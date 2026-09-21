// Richness annotation: for each vibe in the image's pool, decide whether the
// representation's own vibe set (the "target set") already covers it. The
// unit of judgement is a pool vibe, not an atom — which is why this app
// renders item.pool rather than item.atoms.
const state = {
  run: null,
  annotator: null,
  items: [],
  annotations: {},
  index: 0,
  draft: null,
  savedSnapshot: null,
  showLLM: false,
  llmCache: {},
};

const $ = (sel) => document.querySelector(sel);

function draftKey(d) {
  return JSON.stringify(d);
}

function isDirty() {
  if (!state.draft) return false;
  return draftKey(state.draft) !== state.savedSnapshot;
}

function blankDraft(item) {
  return {
    judgements: item.pool.map((p) => ({
      candidate: p.candidate,
      // Unrated starts as "missed": covering a pool vibe is the claim being
      // made, so it has to be asserted rather than defaulted into.
      covered: false,
      reason: "",
    })),
  };
}

function draftFromAnnotation(item, annotation) {
  if (!annotation) return blankDraft(item);
  return {
    judgements: annotation.judgements.map((j) => ({
      candidate: j.candidate,
      covered: j.covered,
      reason: j.reason || "",
    })),
  };
}

// --- setup screen --------------------------------------------------------

async function initSetup() {
  const savedAnnotator = localStorage.getItem("richness_annotator") || "";
  const savedRun = localStorage.getItem("richness_run") || "";
  $("#annotator-input").value = savedAnnotator;

  const res = await fetch("/api/richness/runs");
  const { runs } = await res.json();
  const sel = $("#run-select");
  sel.innerHTML = "";
  for (const r of runs) {
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = r;
    sel.appendChild(opt);
  }
  if (runs.includes(savedRun)) sel.value = savedRun;

  $("#start-btn").addEventListener("click", async () => {
    const annotator = $("#annotator-input").value.trim();
    const run = sel.value;
    if (!annotator || !run) {
      alert("Please enter your name/ID and pick a dataset.");
      return;
    }
    localStorage.setItem("richness_annotator", annotator);
    localStorage.setItem("richness_run", run);
    await startApp(run, annotator);
  });
}

// --- main app --------------------------------------------------------

async function startApp(run, annotator) {
  state.run = run;
  state.annotator = annotator;

  const [datasetRes, annRes] = await Promise.all([
    fetch(`/api/richness/dataset?run=${encodeURIComponent(run)}`),
    fetch(`/api/richness/annotations?run=${encodeURIComponent(run)}&annotator=${encodeURIComponent(annotator)}`),
  ]);
  const dataset = await datasetRes.json();
  const annData = await annRes.json();
  state.items = dataset.items;
  state.annotations = annData.annotations;

  $("#setup").style.display = "none";
  $("#app").classList.add("active");
  $("#who-label").textContent = `${annotator} — ${run}`;

  renderSidebar();

  const firstUnannotated = state.items.findIndex(
    (it) => !(it.image_path in state.annotations)
  );
  loadItem(firstUnannotated === -1 ? 0 : firstUnannotated);
}

function renderSidebar() {
  const ul = $("#item-list");
  ul.innerHTML = "";
  state.items.forEach((item, i) => {
    const li = document.createElement("li");
    li.dataset.index = i;
    const done = item.image_path in state.annotations;
    if (done) li.classList.add("done");
    if (i === state.index) li.classList.add("current");
    const dot = document.createElement("span");
    dot.className = "dot";
    const label = document.createElement("span");
    label.textContent = `${i + 1}. ${item.image_path.split("/").pop()}`;
    li.appendChild(dot);
    li.appendChild(label);
    li.addEventListener("click", () => goTo(i));
    ul.appendChild(li);
  });
  updateProgress();
}

function updateProgress() {
  const total = state.items.length;
  const done = Object.keys(state.annotations).length;
  $("#progress-bar-fill").style.width = `${total ? (done / total) * 100 : 0}%`;
  $("#progress-label").textContent = `${done} / ${total} annotated`;
}

function goTo(i) {
  if (isDirty()) {
    const ok = confirm("You have unsaved changes on this item. Discard and move on?");
    if (!ok) return;
  }
  loadItem(i);
}

function loadItem(i) {
  state.index = i;
  const item = state.items[i];
  const existing = state.annotations[item.image_path];
  state.draft = draftFromAnnotation(item, existing);
  state.savedSnapshot = existing ? draftKey(state.draft) : draftKey(blankDraft(item));

  $("#representation-text").textContent = item.representation;
  $("#image-wrap").style.display = "none";
  $("#toggle-image-btn").textContent = "Show image";

  state.showLLM = false;
  $("#toggle-llm-btn").textContent = "Show LLM judgement";

  renderTarget(item);
  renderPool();
  applyLLMOverlay();
  setSaveStatus(existing ? `Saved (last updated ${new Date(existing.updated_at).toLocaleString()})` : "Not yet saved");

  [...document.querySelectorAll("#item-list li")].forEach((li) =>
    li.classList.toggle("current", Number(li.dataset.index) === i)
  );
  const currentLi = document.querySelector(`#item-list li[data-index="${i}"]`);
  if (currentLi) currentLi.scrollIntoView({ block: "nearest" });
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function renderTarget(item) {
  const row = $("#target-chips");
  row.innerHTML = "";
  const target = item.target || [];
  if (!target.length) {
    row.innerHTML = `<span class="vibe-chip empty">(empty — every pool vibe is missed)</span>`;
    return;
  }
  target.forEach((vibe) => {
    const chip = document.createElement("span");
    chip.className = "vibe-chip";
    chip.textContent = vibe;
    row.appendChild(chip);
  });
}

function renderPool() {
  const container = $("#pool-container");
  container.innerHTML = "";
  state.draft.judgements.forEach((entry, idx) => {
    const card = document.createElement("div");
    card.className = "atom-card";

    const text = document.createElement("div");
    text.className = "atom-text";
    text.innerHTML = `${idx + 1}. ${escapeHtml(entry.candidate)}`;
    card.appendChild(text);

    const toggleRow = document.createElement("div");
    toggleRow.className = "rating-row";
    const toggleGroup = document.createElement("div");
    toggleGroup.className = "toggle-group";

    const yesBtn = document.createElement("button");
    yesBtn.textContent = "Covered";
    const noBtn = document.createElement("button");
    noBtn.textContent = "Missed";

    const badge = document.createElement("span");

    function refreshToggle() {
      yesBtn.classList.toggle("selected", entry.covered === true);
      yesBtn.classList.toggle("yes", entry.covered === true);
      noBtn.classList.toggle("selected", entry.covered === false);
      noBtn.classList.toggle("no", entry.covered === false);
      badge.textContent = entry.covered ? "Covered (redundant)" : "Missed (distinct)";
      badge.className = `verdict-badge ${entry.covered ? "good" : "bad"}`;
    }

    yesBtn.addEventListener("click", () => {
      entry.covered = true;
      refreshToggle();
      setSaveStatus("Unsaved changes");
      applyLLMOverlay();
    });
    noBtn.addEventListener("click", () => {
      entry.covered = false;
      refreshToggle();
      setSaveStatus("Unsaved changes");
      applyLLMOverlay();
    });

    toggleGroup.appendChild(yesBtn);
    toggleGroup.appendChild(noBtn);
    toggleRow.appendChild(toggleGroup);
    toggleRow.appendChild(badge);
    card.appendChild(toggleRow);
    refreshToggle();

    const reasonInput = document.createElement("textarea");
    reasonInput.className = "reason-input";
    reasonInput.placeholder = "Witness test / reason (optional)";
    reasonInput.value = entry.reason || "";
    reasonInput.addEventListener("input", () => {
      entry.reason = reasonInput.value;
      setSaveStatus("Unsaved changes");
    });
    card.appendChild(reasonInput);

    container.appendChild(card);
  });
}

function setSaveStatus(text) {
  $("#save-status").textContent = text;
}

// --- LLM comparison (optional, opt-in per item) --------------------------

async function ensureLLMLoaded(item) {
  if (state.llmCache[item.image_path]) return state.llmCache[item.image_path];
  const res = await fetch(
    `/api/richness/llm_judgement?run=${encodeURIComponent(state.run)}&image_path=${encodeURIComponent(item.image_path)}`
  );
  if (!res.ok) {
    alert("No LLM judgement available for this image.");
    return null;
  }
  const data = await res.json();
  state.llmCache[item.image_path] = data;
  return data;
}

function clearLLMOverlay() {
  $("#llm-summary-panel").style.display = "none";
  document.querySelectorAll(".llm-atom-block").forEach((el) => el.remove());
  document.querySelectorAll(".atom-card").forEach((el) => el.classList.remove("disagree"));
}

function applyLLMOverlay() {
  const item = state.items[state.index];
  const llm = state.llmCache[item.image_path];

  if (!state.showLLM || !llm) {
    clearLLMOverlay();
    return;
  }

  const llmJudgements = llm.judgements || [];
  const cards = document.querySelectorAll("#pool-container .atom-card");
  let disagreeCount = 0;
  let comparedCount = 0;

  cards.forEach((card, idx) => {
    card.querySelectorAll(".llm-atom-block").forEach((el) => el.remove());
    card.classList.remove("disagree");

    const llmEntry = llmJudgements[idx];
    const humanEntry = state.draft.judgements[idx];
    if (!llmEntry) return;
    comparedCount++;

    const llmCovered = llmEntry.verdict === "redundant";
    if (humanEntry.covered !== llmCovered) {
      card.classList.add("disagree");
      disagreeCount++;
    }

    const block = document.createElement("div");
    block.className = "llm-atom-block";
    const badge = document.createElement("span");
    badge.className = `llm-atom-badge ${llmCovered ? "good" : "bad"}`;
    badge.textContent = `LLM: ${llmCovered ? "Covered" : "Missed"}`;
    block.appendChild(badge);
    block.append(llmEntry.witness_test || "");
    card.appendChild(block);
  });

  $("#llm-summary-panel").style.display = "block";
  $("#llm-summary-body").innerHTML = `
    <div class="${disagreeCount === 0 ? "agree" : "disagree"}">
      Pool vibe verdicts: ${comparedCount - disagreeCount}/${comparedCount} agree${disagreeCount ? `, ${disagreeCount} disagreement(s) — highlighted below` : ""}
    </div>
    <div>LLM coverage: ${typeof llm.score === "number" ? llm.score.toFixed(3) : "–"}${
      typeof llm.n_covered === "number" ? ` (${llm.n_covered}/${llm.pool_size} covered)` : ""
    }</div>
  `;
}

async function saveCurrent() {
  const item = state.items[state.index];
  const body = {
    run: state.run,
    annotator: state.annotator,
    image_path: item.image_path,
    judgements: state.draft.judgements.map((j) => ({
      candidate: j.candidate,
      covered: j.covered,
      reason: j.reason ? j.reason.trim() || null : null,
    })),
  };
  const res = await fetch("/api/richness/annotations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    alert("Save failed: " + (await res.text()));
    return false;
  }
  const { saved } = await res.json();
  state.annotations[item.image_path] = saved;
  state.savedSnapshot = draftKey(state.draft);
  setSaveStatus(
    `Saved ✓ ${new Date(saved.updated_at).toLocaleTimeString()} — coverage ${saved.n_covered}/${saved.pool_size} (${saved.score.toFixed(3)})`
  );
  renderSidebar();
  return true;
}

function wireControls() {
  $("#save-btn").addEventListener("click", saveCurrent);

  $("#next-btn").addEventListener("click", () => {
    if (state.index < state.items.length - 1) goTo(state.index + 1);
  });
  $("#prev-btn").addEventListener("click", () => {
    if (state.index > 0) goTo(state.index - 1);
  });

  $("#toggle-image-btn").addEventListener("click", async () => {
    const wrap = $("#image-wrap");
    const img = $("#image-el");
    const item = state.items[state.index];
    if (wrap.style.display === "none" || !wrap.style.display) {
      img.src = `/api/image?path=${encodeURIComponent(item.image_path)}`;
      wrap.style.display = "block";
      $("#toggle-image-btn").textContent = "Hide image";
    } else {
      wrap.style.display = "none";
      $("#toggle-image-btn").textContent = "Show image";
    }
  });

  $("#toggle-llm-btn").addEventListener("click", async () => {
    const item = state.items[state.index];
    if (!state.showLLM) {
      if (
        !(item.image_path in state.annotations) &&
        !confirm(
          "You haven't saved your own rating for this item yet. Seeing the LLM's judgement first may bias your rating. Show anyway?"
        )
      ) {
        return;
      }
      const data = await ensureLLMLoaded(item);
      if (!data) return;
      state.showLLM = true;
    } else {
      state.showLLM = false;
    }
    $("#toggle-llm-btn").textContent = state.showLLM ? "Hide LLM judgement" : "Show LLM judgement";
    applyLLMOverlay();
  });

  $("#toggle-rubric-btn").addEventListener("click", () => {
    const body = $("#rubric-body");
    body.classList.toggle("open");
    $("#toggle-rubric-btn").textContent = body.classList.contains("open")
      ? "Hide evaluation rubric"
      : "Show evaluation rubric";
  });

  $("#switch-btn").addEventListener("click", () => {
    if (isDirty() && !confirm("You have unsaved changes. Leave anyway?")) return;
    location.reload();
  });

  window.addEventListener("beforeunload", (e) => {
    if (isDirty()) {
      e.preventDefault();
      e.returnValue = "";
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.key === "ArrowRight") $("#next-btn").click();
    if (e.key === "ArrowLeft") $("#prev-btn").click();
  });
}

initSetup();
wireControls();
