// Read-only richness viewer: the LLM judge's coverage verdict + witness test
// for every vibe in an image's pool, against the target set the
// representation produced.
const state = {
  run: null,
  items: [],
  index: 0,
  llmCache: {},
  // Missed pool vibes are the interesting half of a richness run, so the
  // covered ones can be folded away.
  showCovered: true,
};

const $ = (sel) => document.querySelector(sel);

// --- setup screen --------------------------------------------------------

async function initSetup() {
  const savedRun = localStorage.getItem("richness_results_run") || "";

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
    const run = sel.value;
    if (!run) {
      alert("Please pick a dataset.");
      return;
    }
    localStorage.setItem("richness_results_run", run);
    await startApp(run);
  });
}

// --- main app --------------------------------------------------------

async function startApp(run) {
  state.run = run;

  const res = await fetch(`/api/richness/dataset?run=${encodeURIComponent(run)}`);
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

  renderTarget(item);
  setSaveStatus("Loading LLM judgement…");
  $("#pool-container").innerHTML = "";
  $("#final-verdict").innerHTML = "";

  const llm = await ensureLLMLoaded(item);
  renderPool(llm);
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
    `/api/richness/llm_judgement?run=${encodeURIComponent(state.run)}&image_path=${encodeURIComponent(item.image_path)}`
  );
  const data = res.ok ? await res.json() : null;
  state.llmCache[item.image_path] = data;
  return data;
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
    row.innerHTML = `<span class="vibe-chip empty">(empty)</span>`;
    return;
  }
  target.forEach((vibe) => {
    const chip = document.createElement("span");
    chip.className = "vibe-chip";
    chip.textContent = vibe;
    row.appendChild(chip);
  });
}

function renderPool(llm) {
  const container = $("#pool-container");
  container.innerHTML = "";
  const judgements = llm?.judgements || [];

  judgements.forEach((j, idx) => {
    const covered = j.verdict === "redundant";
    if (covered && !state.showCovered) return;

    const card = document.createElement("div");
    card.className = "atom-card";
    card.innerHTML = `
      <div class="rating-row">
        <div class="atom-text" style="margin:0;">${idx + 1}. ${escapeHtml(j.candidate)}</div>
        <span class="verdict-badge ${covered ? "good" : "bad"}">${covered ? "Covered" : "Missed"}</span>
      </div>
      <div class="llm-atom-block">${escapeHtml(j.witness_test || "")}</div>
    `;
    container.appendChild(card);
  });

  if (!container.children.length) {
    container.innerHTML = `<div style="color:var(--muted);">${
      judgements.length ? "Every pool vibe was covered." : "No pool judgements recorded for this image."
    }</div>`;
  }
}

function renderFinalVerdict(llm) {
  const el = $("#final-verdict");
  if (!llm) {
    el.innerHTML = `<div style="color:var(--muted);">No LLM judgement recorded for this image.</div>`;
    return;
  }
  const pct = (v) => (typeof v === "number" ? v.toFixed(3) : "–");
  const missed = llm.missed || [];
  el.innerHTML = `
    <div><strong>Coverage:</strong> ${pct(llm.score)}${
      llm.passed === true ? " (passed)" : llm.passed === false ? " (failed)" : ""
    } — ${llm.n_covered ?? "–"}/${llm.pool_size ?? "–"} pool vibes covered</div>
    <div style="color:var(--muted); font-size:0.85rem; margin-top:4px;">
      Support-weighted coverage: ${pct(llm.weighted_coverage)} ·
      target set: ${(llm.target || []).length} vibe(s)
    </div>
    ${
      missed.length
        ? `<div style="margin-top:10px;"><strong>Missed:</strong> <span style="color:var(--bad);">${missed
            .map(escapeHtml)
            .join(" · ")}</span></div>`
        : ""
    }
  `;
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

  $("#toggle-covered-btn").addEventListener("click", () => {
    state.showCovered = !state.showCovered;
    $("#toggle-covered-btn").textContent = state.showCovered
      ? "Hide covered pool vibes"
      : "Show covered pool vibes";
    renderPool(state.llmCache[state.items[state.index].image_path]);
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
