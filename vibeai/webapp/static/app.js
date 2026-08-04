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
    completeness: null,
    claim_independence: null,
    atoms: item.atoms.map((atom) => ({
      atom,
      evaluation: {
        affectiveness: false,
        atomicity: false,
        evidence_preservation: false,
        faithfulness: false,
      },
    })),
  };
}

function draftFromAnnotation(item, annotation) {
  if (!annotation) return blankDraft(item);
  return {
    completeness: annotation.final_verdict.completeness.verdict,
    claim_independence: annotation.final_verdict.claim_independence.verdict,
    atoms: annotation.atomic_judgement.map((aj) => ({
      atom: aj.atom,
      evaluation: { ...aj.evaluation },
    })),
  };
}

// --- setup screen --------------------------------------------------------

async function initSetup() {
  const savedAnnotator = localStorage.getItem("annotator") || "";
  const savedRun = localStorage.getItem("run") || "";
  $("#annotator-input").value = savedAnnotator;

  const res = await fetch("/api/runs");
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
    localStorage.setItem("annotator", annotator);
    localStorage.setItem("run", run);
    await startApp(run, annotator);
  });
}

// --- main app --------------------------------------------------------

async function startApp(run, annotator) {
  state.run = run;
  state.annotator = annotator;

  const [datasetRes, annRes] = await Promise.all([
    fetch(`/api/dataset?run=${encodeURIComponent(run)}`),
    fetch(`/api/annotations?run=${encodeURIComponent(run)}&annotator=${encodeURIComponent(annotator)}`),
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

  renderScale("completeness");
  renderScale("claim_independence");
  renderAtoms();
  applyLLMOverlay();
  setSaveStatus(existing ? `Saved (last updated ${new Date(existing.updated_at).toLocaleString()})` : "Not yet saved");

  [...document.querySelectorAll("#item-list li")].forEach((li) =>
    li.classList.toggle("current", Number(li.dataset.index) === i)
  );
  const currentLi = document.querySelector(`#item-list li[data-index="${i}"]`);
  if (currentLi) currentLi.scrollIntoView({ block: "nearest" });
}

function renderScale(field) {
  const container = document.querySelector(`.scale[data-field="${field}"]`);
  container.innerHTML = "";
  for (let v = 1; v <= 5; v++) {
    const btn = document.createElement("button");
    btn.textContent = v;
    if (state.draft[field] === v) btn.classList.add("selected");
    btn.addEventListener("click", () => {
      state.draft[field] = v;
      renderScale(field);
      setSaveStatus("Unsaved changes");
      applyLLMOverlay();
    });
    container.appendChild(btn);
  }
}

const CRITERIA = [
  ["affectiveness", "Affective"],
  ["atomicity", "Atomic"],
  ["evidence_preservation", "Evidence preserved"],
  ["faithfulness", "Faithful"],
];

function atomVerdict(evaluation) {
  return CRITERIA.every(([key]) => evaluation[key]) ? "Good" : "Bad";
}

function renderAtoms() {
  const container = $("#atoms-container");
  container.innerHTML = "";
  state.draft.atoms.forEach((entry, idx) => {
    const card = document.createElement("div");
    card.className = "atom-card";

    const text = document.createElement("div");
    text.className = "atom-text";
    text.textContent = `${idx + 1}. ${entry.atom}`;
    card.appendChild(text);

    const criteria = document.createElement("div");
    criteria.className = "criteria";

    for (const [key, label] of CRITERIA) {
      const lbl = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = entry.evaluation[key];
      cb.addEventListener("change", () => {
        entry.evaluation[key] = cb.checked;
        badge.textContent = atomVerdict(entry.evaluation);
        badge.className = `verdict-badge ${atomVerdict(entry.evaluation).toLowerCase()}`;
        setSaveStatus("Unsaved changes");
        applyLLMOverlay();
      });
      lbl.appendChild(cb);
      lbl.append(label);
      criteria.appendChild(lbl);
    }

    const badge = document.createElement("span");
    badge.className = `verdict-badge ${atomVerdict(entry.evaluation).toLowerCase()}`;
    badge.textContent = atomVerdict(entry.evaluation);
    criteria.appendChild(badge);

    card.appendChild(criteria);
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
    `/api/llm_judgement?run=${encodeURIComponent(state.run)}&image_path=${encodeURIComponent(item.image_path)}`
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
  for (const id of ["completeness-llm-note", "claim-independence-llm-note"]) {
    const el = document.getElementById(id);
    el.textContent = "";
    el.classList.remove("diff");
  }
  document.querySelectorAll(".llm-atom-block").forEach((el) => el.remove());
  document.querySelectorAll(".atom-card").forEach((el) => el.classList.remove("disagree"));
  document.querySelectorAll(".criteria label.diff").forEach((el) => el.classList.remove("diff"));
}

function applyLLMOverlay() {
  const item = state.items[state.index];
  const llm = state.llmCache[item.image_path];

  if (!state.showLLM || !llm) {
    clearLLMOverlay();
    return;
  }

  const fv = llm.final_verdict;
  const humanCompleteness = state.draft.completeness;
  const humanIndependence = state.draft.claim_independence;

  const compNote = $("#completeness-llm-note");
  compNote.textContent = `LLM: ${fv.completeness.verdict} — ${fv.completeness.reason}`;
  compNote.classList.toggle(
    "diff",
    humanCompleteness !== null && humanCompleteness !== fv.completeness.verdict
  );

  const indepNote = $("#claim-independence-llm-note");
  indepNote.textContent = `LLM: ${fv.claim_independence.verdict} — ${fv.claim_independence.reason}`;
  indepNote.classList.toggle(
    "diff",
    humanIndependence !== null && humanIndependence !== fv.claim_independence.verdict
  );

  const llmAtoms = llm.atomic_judgement || [];
  const cards = document.querySelectorAll(".atom-card");
  let disagreeCount = 0;
  let comparedCount = 0;

  cards.forEach((card, idx) => {
    card.querySelectorAll(".llm-atom-block").forEach((el) => el.remove());
    card.classList.remove("disagree");
    card.querySelectorAll(".criteria label").forEach((el) => el.classList.remove("diff"));

    const llmEntry = llmAtoms[idx];
    const humanEntry = state.draft.atoms[idx];
    if (!llmEntry) return;
    comparedCount++;

    const humanVerdict = atomVerdict(humanEntry.evaluation);
    if (humanVerdict !== llmEntry.verdict) {
      card.classList.add("disagree");
      disagreeCount++;
    }

    const labels = card.querySelectorAll(".criteria label");
    CRITERIA.forEach(([key], ci) => {
      if (humanEntry.evaluation[key] !== llmEntry.evaluation[key]) {
        labels[ci].classList.add("diff");
      }
    });

    const block = document.createElement("div");
    block.className = "llm-atom-block";
    const badge = document.createElement("span");
    badge.className = `llm-atom-badge ${llmEntry.verdict.toLowerCase()}`;
    badge.textContent = `LLM: ${llmEntry.verdict}`;
    block.appendChild(badge);
    block.append(llmEntry.reason || "");
    card.appendChild(block);
  });

  const completenessDiff =
    humanCompleteness !== null ? Math.abs(humanCompleteness - fv.completeness.verdict) : null;
  const independenceDiff =
    humanIndependence !== null
      ? Math.abs(humanIndependence - fv.claim_independence.verdict)
      : null;
  const atomQualityVerdict = fv.atom_quality.verdict;

  $("#llm-summary-panel").style.display = "block";
  $("#llm-summary-body").innerHTML = `
    <div class="${completenessDiff === null ? "" : completenessDiff === 0 ? "agree" : "disagree"}">
      Completeness: you=${humanCompleteness ?? "–"} vs LLM=${fv.completeness.verdict}${completenessDiff ? ` (Δ${completenessDiff})` : ""}
    </div>
    <div class="${independenceDiff === null ? "" : independenceDiff === 0 ? "agree" : "disagree"}">
      Claim independence: you=${humanIndependence ?? "–"} vs LLM=${fv.claim_independence.verdict}${independenceDiff ? ` (Δ${independenceDiff})` : ""}
    </div>
    <div class="${disagreeCount === 0 ? "agree" : "disagree"}">
      Atom verdicts: ${comparedCount - disagreeCount}/${comparedCount} agree${disagreeCount ? `, ${disagreeCount} disagreement(s) — highlighted below` : ""}
    </div>
    <div>LLM atom quality: ${typeof atomQualityVerdict === "number" ? atomQualityVerdict.toFixed(2) : atomQualityVerdict} — ${fv.atom_quality.reason}</div>
  `;
}

async function saveCurrent() {
  const item = state.items[state.index];
  if (state.draft.completeness === null || state.draft.claim_independence === null) {
    alert("Please rate both Completeness and Claim Independence before saving.");
    return false;
  }
  const body = {
    run: state.run,
    annotator: state.annotator,
    image_path: item.image_path,
    atomic_judgement: state.draft.atoms.map((a) => ({
      atom: a.atom,
      evaluation: a.evaluation,
    })),
    completeness: state.draft.completeness,
    claim_independence: state.draft.claim_independence,
  };
  const res = await fetch("/api/annotations", {
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
  setSaveStatus(`Saved ✓ ${new Date(saved.updated_at).toLocaleTimeString()}`);
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
    if (e.target.tagName === "INPUT") return;
    if (e.key === "ArrowRight") $("#next-btn").click();
    if (e.key === "ArrowLeft") $("#prev-btn").click();
  });
}

initSetup();
wireControls();
