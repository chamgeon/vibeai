const state = {
  run: null,
  annotator: null,
  items: [],
  visible: [], // indices into items, after the pair filter
  annotations: {},
  index: 0, // into items
  draft: null, // {choice, reason}
  savedSnapshot: null,
  showLLM: false,
  llmCache: {},
};

const $ = (sel) => document.querySelector(sel);

function draftKey(d) {
  return JSON.stringify(d);
}

function isDirty() {
  return state.draft !== null && draftKey(state.draft) !== state.savedSnapshot;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s ?? "";
  return div.innerHTML;
}

function choiceText(choice) {
  return choice === "tie" ? "Tie" : choice ? `Set ${choice} richer` : "—";
}

// --- setup screen --------------------------------------------------------

async function initSetup() {
  $("#annotator-input").value = localStorage.getItem("rich_annotator") || "";
  const savedRun = localStorage.getItem("rich_run") || "";

  const { runs } = await (await fetch("/api/richness/runs")).json();
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
    localStorage.setItem("rich_annotator", annotator);
    localStorage.setItem("rich_run", run);
    await startApp(run, annotator);
  });
}

// --- main app --------------------------------------------------------

async function startApp(run, annotator) {
  state.run = run;
  state.annotator = annotator;

  const q = (params) => new URLSearchParams(params).toString();
  const [dataset, annData] = await Promise.all([
    fetch(`/api/richness/dataset?${q({ run })}`).then((r) => r.json()),
    fetch(`/api/richness/annotations?${q({ run, annotator })}`).then((r) => r.json()),
  ]);
  state.items = dataset.items;
  state.annotations = annData.annotations;

  $("#setup").style.display = "none";
  $("#app").classList.add("active");
  $("#who-label").textContent = `${annotator} — ${run}`;

  const pairs = [...new Set(state.items.map((it) => it.pair.join(" vs ")))];
  const filter = $("#pair-filter");
  filter.innerHTML = `<option value="">All pairs (${state.items.length})</option>` +
    pairs.map((p) => {
      const n = state.items.filter((it) => it.pair.join(" vs ") === p).length;
      return `<option value="${escapeHtml(p)}">${escapeHtml(p)} (${n})</option>`;
    }).join("");
  filter.value = localStorage.getItem("rich_pair_filter") || "";
  if (![...filter.options].some((o) => o.value === filter.value)) filter.value = "";

  state.showLLM = localStorage.getItem("rich_show_llm") === "1";
  $("#llm-switch").checked = state.showLLM;

  applyFilter();
}

function applyFilter() {
  const pair = $("#pair-filter").value;
  state.visible = state.items
    .map((it, i) => i)
    .filter((i) => !pair || state.items[i].pair.join(" vs ") === pair);
  renderSidebar();
  const firstOpen = state.visible.find((i) => !(state.items[i].comparison_id in state.annotations));
  loadItem(firstOpen ?? state.visible[0] ?? 0);
}

function renderSidebar() {
  const ul = $("#item-list");
  ul.innerHTML = "";
  state.visible.forEach((i, pos) => {
    const item = state.items[i];
    const li = document.createElement("li");
    li.dataset.index = i;
    if (item.comparison_id in state.annotations) li.classList.add("done");
    if (i === state.index) li.classList.add("current");
    li.innerHTML = `<span class="dot"></span><span>${pos + 1}. ${escapeHtml(item.pair.join(" vs "))} · ${escapeHtml(item.image_path.split("/").pop())}</span>`;
    li.addEventListener("click", () => goTo(i));
    ul.appendChild(li);
  });
  updateProgress();
}

function updateProgress() {
  const total = state.visible.length;
  const done = state.visible.filter((i) => state.items[i].comparison_id in state.annotations).length;
  $("#progress-bar-fill").style.width = `${total ? (done / total) * 100 : 0}%`;
  $("#progress-label").textContent = `${done} / ${total} annotated`;
}

function goTo(i) {
  if (isDirty() && !confirm("You have unsaved changes on this comparison. Discard and move on?")) return;
  loadItem(i);
}

function step(delta) {
  const pos = state.visible.indexOf(state.index);
  const next = state.visible[pos + delta];
  if (next !== undefined) goTo(next);
}

function renderSet(n, vibes) {
  $(`#set-${n}-count`).textContent = `· ${vibes.length} vibe${vibes.length === 1 ? "" : "s"}`;
  $(`#set-${n}-list`).innerHTML = vibes.length
    ? vibes.map((v) => `<li>${escapeHtml(v)}</li>`).join("")
    : `<li class="empty">(empty)</li>`;
}

function loadItem(i) {
  state.index = i;
  const item = state.items[i];
  if (!item) return;
  const existing = state.annotations[item.comparison_id];
  state.draft = { choice: existing?.choice ?? null, reason: existing?.reason ?? "" };
  state.savedSnapshot = existing ? draftKey(state.draft) : draftKey({ choice: null, reason: "" });

  $("#item-heading").textContent = `Image · ${item.image_path.split("/").pop()}`;
  $("#image-el").src = `/api/image?path=${encodeURIComponent(item.image_path)}`;
  renderSet(1, item.set_1);
  renderSet(2, item.set_2);
  $("#reason-input").value = state.draft.reason;
  renderChoice();
  setSaveStatus(existing ? `Saved (last updated ${new Date(existing.updated_at).toLocaleString()})` : "Not yet saved");

  document.querySelectorAll("#item-list li").forEach((li) =>
    li.classList.toggle("current", Number(li.dataset.index) === i)
  );
  document.querySelector(`#item-list li[data-index="${i}"]`)?.scrollIntoView({ block: "nearest" });

  renderLLM();
  document.querySelector("#main").scrollTop = 0;
}

function renderChoice() {
  document.querySelectorAll(".choice-group button").forEach((btn) =>
    btn.classList.toggle("selected", btn.dataset.choice === state.draft.choice)
  );
  for (const n of ["1", "2"]) {
    $(`#set-${n}-col`).classList.toggle("chosen", state.draft.choice === n);
  }
}

function setChoice(choice) {
  state.draft.choice = choice;
  renderChoice();
  setSaveStatus("Unsaved changes");
  renderLLM();
}

function setSaveStatus(text) {
  $("#save-status").textContent = text;
}

// --- LLM judgement (on/off switch, persists across comparisons) ----------

async function ensureLLMLoaded(item) {
  if (item.comparison_id in state.llmCache) return state.llmCache[item.comparison_id];
  const params = new URLSearchParams({ run: state.run, comparison_id: item.comparison_id });
  const res = await fetch(`/api/richness/llm_judgement?${params}`);
  const data = res.ok ? await res.json() : null;
  state.llmCache[item.comparison_id] = data;
  return data;
}

function renderOrder(order, idx) {
  const facets = order.image_facets.map((f) => `<li>${escapeHtml(f)}</li>`).join("");
  return `
    <details class="llm-order">
      <summary>
        Order ${idx + 1}: Set ${order.set_shown_as_a} shown first →
        <strong>${escapeHtml(choiceText(order.winner))}</strong>
        <span class="hint">(the judge's "Set A" = Set ${order.set_shown_as_a}, "Set B" = Set ${order.set_shown_as_a === "1" ? "2" : "1"})</span>
      </summary>
      <div class="llm-order-body">
        <div class="llm-label">Image facets the judge saw</div>
        <ul>${facets}</ul>
        <div class="llm-label">Set 1</div><p>${escapeHtml(order.analysis["1"])}</p>
        <div class="llm-label">Set 2</div><p>${escapeHtml(order.analysis["2"])}</p>
        <div class="llm-label">Comparison</div><p>${escapeHtml(order.comparison)}</p>
      </div>
    </details>`;
}

async function renderLLM() {
  const panel = $("#llm-panel");
  if (!state.showLLM) {
    panel.style.display = "none";
    return;
  }
  const item = state.items[state.index];
  const llm = await ensureLLMLoaded(item);
  if (item !== state.items[state.index]) return; // moved on while loading
  panel.style.display = "block";
  if (!llm) {
    $("#llm-body").innerHTML = `<div class="hint">No LLM judgement for this comparison.</div>`;
    return;
  }

  const saved = state.annotations[item.comparison_id];
  const human = state.draft.choice;
  let agreement = "";
  if (human) {
    const agree = human === llm.winner;
    agreement = `<div class="${agree ? "agree" : "disagree"}">You: ${choiceText(human)} — ${agree ? "agrees with" : "differs from"} the LLM</div>`;
  } else {
    agreement = `<div class="hint">You haven't chosen yet — the LLM's verdict may anchor your own.</div>`;
  }
  // System names only once your own judgement is saved, so the switch can
  // stay on without telling you which system wrote which set.
  const systems = saved
    ? `<div class="hint">Set 1 = <strong>${escapeHtml(llm.systems["1"])}</strong>, Set 2 = <strong>${escapeHtml(llm.systems["2"])}</strong></div>`
    : `<div class="hint">Systems are revealed after you save.</div>`;

  $("#llm-body").innerHTML = `
    <div id="llm-summary-body">
      <div><span class="verdict-badge ${llm.winner === "tie" ? "" : "good"}">LLM: ${escapeHtml(choiceText(llm.winner))}</span>
        <span class="hint">${llm.consistent ? "both orders agreed" : "orders disagreed"}</span></div>
      ${agreement}
      ${systems}
    </div>
    ${llm.orders.map(renderOrder).join("")}
  `;
}

async function saveCurrent() {
  const item = state.items[state.index];
  if (!state.draft.choice) {
    alert("Pick Set 1, Set 2 or Tie first.");
    return false;
  }
  const res = await fetch("/api/richness/annotations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run: state.run,
      annotator: state.annotator,
      comparison_id: item.comparison_id,
      choice: state.draft.choice,
      reason: state.draft.reason.trim() || null,
    }),
  });
  if (!res.ok) {
    alert("Save failed: " + (await res.text()));
    return false;
  }
  const { saved } = await res.json();
  state.annotations[item.comparison_id] = saved;
  state.savedSnapshot = draftKey(state.draft);
  setSaveStatus(`Saved ✓ ${new Date(saved.updated_at).toLocaleTimeString()}`);
  renderSidebar();
  renderLLM();
  return true;
}

function wireControls() {
  $("#save-btn").addEventListener("click", saveCurrent);
  $("#next-btn").addEventListener("click", () => step(1));
  $("#prev-btn").addEventListener("click", () => step(-1));

  document.querySelectorAll(".choice-group button").forEach((btn) =>
    btn.addEventListener("click", () => setChoice(btn.dataset.choice))
  );
  $("#reason-input").addEventListener("input", (e) => {
    state.draft.reason = e.target.value;
    setSaveStatus("Unsaved changes");
  });

  $("#pair-filter").addEventListener("change", (e) => {
    if (isDirty() && !confirm("You have unsaved changes. Discard and switch filter?")) {
      e.target.value = localStorage.getItem("rich_pair_filter") || "";
      return;
    }
    localStorage.setItem("rich_pair_filter", e.target.value);
    applyFilter();
  });

  $("#llm-switch").addEventListener("change", (e) => {
    state.showLLM = e.target.checked;
    localStorage.setItem("rich_show_llm", state.showLLM ? "1" : "0");
    renderLLM();
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
    if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
    if (!$("#app").classList.contains("active")) return;
    if (e.key === "ArrowRight") step(1);
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "1" || e.key === "2") setChoice(e.key);
    else if (e.key === "t") setChoice("tie");
    else if (e.key === "s") saveCurrent();
  });
}

initSetup();
wireControls();
