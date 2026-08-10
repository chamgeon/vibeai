const state = {
  run: null,
  items: [],
  index: 0,
  llmCache: {},
};

const $ = (sel) => document.querySelector(sel);

// --- setup screen --------------------------------------------------------

async function initSetup() {
  const savedRun = localStorage.getItem("plaus_results_run") || "";

  const res = await fetch("/api/plausibility/runs");
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
    localStorage.setItem("plaus_results_run", run);
    await startApp(run);
  });
}

// --- main app --------------------------------------------------------

async function startApp(run) {
  state.run = run;

  const res = await fetch(`/api/plausibility/dataset?run=${encodeURIComponent(run)}`);
  const dataset = await res.json();
  state.items = dataset.items;
  state.llmCache = {};

  $("#setup").style.display = "none";
  $("#app").classList.add("active");
  $("#who-label").textContent = run;

  renderSidebar();
  loadItem(0);
}

function renderSidebar() {
  const ul = $("#item-list");
  ul.innerHTML = "";
  state.items.forEach((item, i) => {
    const li = document.createElement("li");
    li.dataset.index = i;
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
  renderAtoms(llm);
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
    `/api/plausibility/llm_judgement?run=${encodeURIComponent(state.run)}&image_path=${encodeURIComponent(item.image_path)}`
  );
  const data = res.ok ? await res.json() : null;
  state.llmCache[item.image_path] = data;
  return data;
}

function renderCheck(title, check) {
  if (!check) return "";
  const badge = `<span class="verdict-badge ${check.verdict ? "good" : "bad"}">${check.verdict ? "Pass" : "Fail"}</span>`;
  let extra = "";
  if ("supporting_evidence" in check) {
    if (check.supporting_evidence?.length) {
      extra += `<ul>${check.supporting_evidence.map((e) => `<li>+ ${escapeHtml(e)}</li>`).join("")}</ul>`;
    }
    if (check.contradicting_evidence?.length) {
      extra += `<ul>${check.contradicting_evidence.map((e) => `<li>− ${escapeHtml(e)}</li>`).join("")}</ul>`;
    }
  }
  return `
    <div class="check-block">
      <div class="check-title">${title}${badge}</div>
      <div style="color:var(--muted);">${escapeHtml(check.reasoning || "")}</div>
      ${extra}
    </div>
  `;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function renderAtoms(llm) {
  const container = $("#atoms-container");
  container.innerHTML = "";
  const llmAtoms = llm?.atoms || [];

  llmAtoms.forEach((a, idx) => {
    const card = document.createElement("div");
    card.className = "atom-card";

    const maxScore = a.type === "vibe_only" ? 1 : 2;
    card.innerHTML = `
      <div class="atom-type-badge">${a.type.replace("_", " ")}</div>
      <div class="atom-text">${idx + 1}. ${escapeHtml(a.atom)}</div>
      <div class="atom-meta">
        Stated vibe: <strong>${escapeHtml(a.stated_vibe || "")}</strong>
        ${a.stated_evidence ? `<br/>Stated evidence: ${a.stated_evidence.map(escapeHtml).join("; ")}` : ""}
      </div>
      <span class="verdict-badge ${a.score > 0 ? "good" : "bad"}">Score: ${a.score} / ${maxScore}</span>
      ${renderCheck("Evidence presence check", a.evidence_presence_check)}
      ${renderCheck("Direct check", a.direct_check)}
      ${renderCheck("Mapping check", a.mapping_check)}
    `;

    container.appendChild(card);
  });
}

function renderFinalVerdict(llm) {
  const el = $("#final-verdict");
  if (!llm) {
    el.innerHTML = `<div style="color:var(--muted);">No LLM judgement recorded for this image.</div>`;
    return;
  }
  const scoreLine =
    typeof llm.score === "number"
      ? `<div><strong>Overall score:</strong> ${llm.score.toFixed(3)}${
          llm.passed === true ? " (passed)" : llm.passed === false ? " (failed)" : ""
        }</div>`
      : "";
  el.innerHTML = scoreLine;
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
