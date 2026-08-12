const state = {
  run: null,
  items: [],
  index: 0,
  llmCache: {},
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

function renderAtoms(item, llm) {
  const container = $("#atoms-container");
  container.innerHTML = "";
  const llmAtoms = llm?.atomic_judgement || [];

  item.atoms.forEach((atom, idx) => {
    const card = document.createElement("div");
    card.className = "atom-card";

    const text = document.createElement("div");
    text.className = "atom-text";
    text.textContent = `${idx + 1}. ${atom}`;
    card.appendChild(text);

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
