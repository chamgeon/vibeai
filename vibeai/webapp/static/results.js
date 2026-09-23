const state = {
  run: null,
  items: [],
  index: 0,
  llmCache: {},
  // Evidence/vibe split is off by default: only structured (v2) runs carry
  // atom_details, so the toggle is disabled entirely for runs without it.
  showSplit: false,
};

const $ = (sel) => document.querySelector(sel);

// --- setup screen --------------------------------------------------------

async function initSetup() {
  const savedRun = localStorage.getItem("results_run") || "";

  const res = await fetch("/api/decomposition_quality/runs");
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
    const run = sel.value;
    if (!run) {
      alert("Please pick a dataset.");
      return;
    }
    localStorage.setItem("results_run", run);
    await startApp(run);
  });
}

// --- main app --------------------------------------------------------

async function startApp(run) {
  state.run = run;

  const res = await fetch(`/api/decomposition_quality/dataset?run=${encodeURIComponent(run)}`);
  const dataset = await res.json();
  state.items = dataset.items;
  state.llmCache = {};

  $("#setup").style.display = "none";
  $("#app").classList.add("active");
  $("#who-label").textContent = run;

  renderSidebar();
  loadItem(0);

  await Promise.all(state.items.map((item) => ensureLLMLoaded(item)));
  renderSidebar();
}

function renderSidebar() {
  const ul = $("#item-list");
  ul.innerHTML = "";
  state.items.forEach((item, i) => {
    const li = document.createElement("li");
    li.dataset.index = i;
    if (i === state.index) li.classList.add("current");
    const llm = state.llmCache[item.image_path];
    if (llm && llm.passed === false) li.classList.add("failed");
    const dot = document.createElement("span");
    dot.className = "dot";
    const label = document.createElement("span");
    label.textContent = `${i + 1}. ${item.image_path.split("/").pop()}`;
    li.appendChild(dot);
    li.appendChild(label);
    li.addEventListener("click", () => goTo(i));
    ul.appendChild(li);
  });
}

function goTo(i) {
  loadItem(i);
}

async function loadItem(i) {
  state.index = i;
  const item = state.items[i];

  $("#representation-text").textContent = item.representation;
  $("#image-wrap").style.display = "none";
  $("#toggle-image-btn").textContent = "Show image";
  const img = $("#image-el");
  img.src = `/api/image?path=${encodeURIComponent(item.image_path)}`;

  setSaveStatus("Loading LLM judgement…");
  $("#atoms-container").innerHTML = "";
  $("#final-verdict").innerHTML = "";

  const llm = await ensureLLMLoaded(item);
  const btn = $("#toggle-split-btn");
  const hasSplit = Boolean(atomDetailsFor(llm));
  btn.disabled = !hasSplit;
  btn.title = hasSplit
    ? ""
    : "This run's decomposition prompt didn't emit an evidence/vibe split (v2 only).";
  renderAtoms(item, llm);
  renderFinalVerdict(llm);
  setSaveStatus(llm ? "" : "No LLM judgement recorded for this image.");

  [...document.querySelectorAll("#item-list li")].forEach((li) =>
    li.classList.toggle("current", Number(li.dataset.index) === i)
  );
  const currentLi = document.querySelector(`#item-list li[data-index="${i}"]`);
  if (currentLi) currentLi.scrollIntoView({ block: "nearest" });
}

async function ensureLLMLoaded(item) {
  if (item.image_path in state.llmCache) return state.llmCache[item.image_path];
  const res = await fetch(
    `/api/decomposition_quality/llm_judgement?run=${encodeURIComponent(state.run)}&image_path=${encodeURIComponent(item.image_path)}`
  );
  const data = res.ok ? await res.json() : null;
  state.llmCache[item.image_path] = data;
  return data;
}

function chip(text, cls) {
  const el = document.createElement("span");
  el.className = `chip ${cls}`;
  el.textContent = text;
  return el;
}

// atom_details is emitted only by structured decomposition prompts (v2) and
// rides along on the llm_judgement payload, which spreads the run record's
// whole `details` dict. Guard on length so a mismatched record can't pair an
// atom with someone else's split.
function atomDetailsFor(llm) {
  const details = llm?.atom_details;
  if (!Array.isArray(details)) return null;
  return details;
}

function buildSplitBlock(detail) {
  const block = document.createElement("div");
  block.className = "atom-split";

  const evidenceRow = document.createElement("div");
  evidenceRow.className = "split-row";
  const evidenceLabel = document.createElement("span");
  evidenceLabel.className = "split-label";
  evidenceLabel.textContent = "Evidence";
  evidenceRow.appendChild(evidenceLabel);
  if (Array.isArray(detail.evidence) && detail.evidence.length) {
    for (const cue of detail.evidence) evidenceRow.appendChild(chip(cue, "evidence"));
  } else {
    evidenceRow.appendChild(chip("none stated", "none"));
  }
  block.appendChild(evidenceRow);

  const vibeRow = document.createElement("div");
  vibeRow.className = "split-row";
  const vibeLabel = document.createElement("span");
  vibeLabel.className = "split-label";
  vibeLabel.textContent = "Vibe";
  vibeRow.appendChild(vibeLabel);
  vibeRow.appendChild(chip(detail.vibe || "—", "vibe"));
  block.appendChild(vibeRow);

  return block;
}

function renderAtoms(item, llm) {
  const container = $("#atoms-container");
  container.innerHTML = "";
  const llmAtoms = llm?.atomic_judgement || [];
  const details = atomDetailsFor(llm);
  const paired = details && details.length === item.atoms.length ? details : null;

  item.atoms.forEach((atom, idx) => {
    const card = document.createElement("div");
    card.className = "atom-card";

    const text = document.createElement("div");
    text.className = "atom-text";
    text.textContent = `${idx + 1}. ${atom}`;

    const detail = paired ? paired[idx] : null;
    if (detail && state.showSplit) {
      const badge = document.createElement("span");
      badge.className = "atom-type-badge";
      badge.textContent = detail.type === "evidence_backed" ? "evidence-backed" : "vibe-only";
      text.appendChild(badge);
    }
    card.appendChild(text);

    if (detail && state.showSplit) card.appendChild(buildSplitBlock(detail));

    const llmEntry = llmAtoms[idx];
    if (llmEntry) {
      const badge = document.createElement("span");
      badge.className = `verdict-badge ${llmEntry.verdict.toLowerCase()}`;
      badge.textContent = `LLM: ${llmEntry.verdict}`;
      card.appendChild(badge);

      const block = document.createElement("div");
      block.className = "llm-atom-block";
      block.textContent = llmEntry.reason || "";
      card.appendChild(block);
    }

    container.appendChild(card);
  });
}

function labelize(key) {
  return key
    .split("_")
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

function renderFinalVerdict(llm) {
  const el = $("#final-verdict");
  if (!llm || !llm.final_verdict) {
    el.innerHTML = `<div style="color:var(--muted);">No LLM judgement recorded for this image.</div>`;
    return;
  }
  const fv = llm.final_verdict;
  const scoreLine =
    typeof llm.score === "number"
      ? `<div style="margin-bottom:10px;"><strong>Overall score:</strong> ${llm.score.toFixed(3)}${
          llm.passed === true ? " (passed)" : llm.passed === false ? " (failed)" : ""
        }</div>`
      : "";

  const sections = Object.entries(fv)
    .map(([key, val]) => {
      if (val == null || typeof val !== "object") return "";
      const verdict = typeof val.verdict === "number" ? val.verdict.toFixed(2).replace(/\.00$/, "") : val.verdict;
      const countSuffix =
        "good_atom_count" in val && "total_atom_count" in val
          ? ` (${val.good_atom_count}/${val.total_atom_count} good atoms)`
          : "";
      return `
        <div style="margin-bottom:14px;">
          <strong>${labelize(key)}:</strong> ${verdict ?? "–"} / 5${countSuffix}
          <div style="color:var(--muted); font-size:0.88rem; margin-top:4px;">${val.reason || ""}</div>
        </div>
      `;
    })
    .join("");

  el.innerHTML = `${scoreLine}${sections}`;
}

function setSaveStatus(text) {
  $("#save-status").textContent = text;
}

function wireControls() {
  $("#next-btn").addEventListener("click", () => {
    if (state.index < state.items.length - 1) goTo(state.index + 1);
  });
  $("#prev-btn").addEventListener("click", () => {
    if (state.index > 0) goTo(state.index - 1);
  });

  $("#toggle-image-btn").addEventListener("click", () => {
    const wrap = $("#image-wrap");
    if (wrap.style.display === "none" || !wrap.style.display) {
      wrap.style.display = "block";
      $("#toggle-image-btn").textContent = "Hide image";
    } else {
      wrap.style.display = "none";
      $("#toggle-image-btn").textContent = "Show image";
    }
  });

  $("#toggle-split-btn").addEventListener("click", () => {
    state.showSplit = !state.showSplit;
    $("#toggle-split-btn").textContent = state.showSplit
      ? "Hide evidence / vibe"
      : "Show evidence / vibe";
    const item = state.items[state.index];
    renderAtoms(item, state.llmCache[item.image_path]);
  });

  $("#switch-btn").addEventListener("click", () => {
    location.reload();
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "ArrowRight") $("#next-btn").click();
    if (e.key === "ArrowLeft") $("#prev-btn").click();
  });
}

initSetup();
wireControls();
