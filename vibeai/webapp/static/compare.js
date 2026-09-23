const state = {
  metric: "plausibility",
  runA: null,
  runB: null,
  rows: [],
  summary: null,
  filter: "all",
  sort: "delta",
  expanded: null,
  detailCache: {},
};

const $ = (sel) => document.querySelector(sel);
const fmt = (v, digits = 2) => (v === null || v === undefined ? "—" : v.toFixed(digits));
const signed = (v) => (v === null || v === undefined ? "—" : (v >= 0 ? "+" : "") + v.toFixed(2));

// Status is never carried by colour alone: every light ships with this word
// beside it (the red/green pair separates by only ~5.7 dE under deuteranopia).
const STATUS_LABEL = {
  regressed: "regressed",
  improved: "improved",
  tied: "no change",
  missing: "not in both",
};

// --- setup ---------------------------------------------------------------

async function loadRuns(metric) {
  const res = await fetch(`/api/${metric}/runs`);
  const { runs } = await res.json();
  for (const [sel, preferred] of [
    ["#run-a-select", localStorage.getItem("cmp_run_a")],
    ["#run-b-select", localStorage.getItem("cmp_run_b")],
  ]) {
    const el = $(sel);
    el.innerHTML = "";
    for (const r of runs) {
      const opt = document.createElement("option");
      opt.value = r;
      opt.textContent = r;
      el.appendChild(opt);
    }
    if (runs.includes(preferred)) el.value = preferred;
  }
  // Sensible default when nothing is remembered: oldest as reference, newest
  // under test, which is the usual "did my change help?" direction.
  if (runs.length >= 2) {
    if (!localStorage.getItem("cmp_run_a")) $("#run-a-select").value = runs[0];
    if (!localStorage.getItem("cmp_run_b")) $("#run-b-select").value = runs[runs.length - 1];
  }
}

async function initSetup() {
  const metricSel = $("#metric-select");
  metricSel.value = localStorage.getItem("cmp_metric") || "plausibility";
  await loadRuns(metricSel.value);
  metricSel.addEventListener("change", () => loadRuns(metricSel.value));

  $("#start-btn").addEventListener("click", async () => {
    const metric = metricSel.value;
    const runA = $("#run-a-select").value;
    const runB = $("#run-b-select").value;
    if (!runA || !runB) return alert("Pick two runs.");
    if (runA === runB) return alert("Pick two different runs.");
    localStorage.setItem("cmp_metric", metric);
    localStorage.setItem("cmp_run_a", runA);
    localStorage.setItem("cmp_run_b", runB);
    await startApp(metric, runA, runB);
  });

  $("#switch-btn").addEventListener("click", () => {
    $("#compare-app").classList.remove("active");
    $("#setup").style.display = "";
  });
  $("#sort-select").addEventListener("change", (e) => {
    state.sort = e.target.value;
    renderTable();
  });
}

// --- main ----------------------------------------------------------------

async function startApp(metric, runA, runB) {
  const res = await fetch(
    `/api/compare?metric=${encodeURIComponent(metric)}` +
      `&run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`
  );
  if (!res.ok) return alert(`Could not load comparison: ${res.status}`);
  const data = await res.json();

  Object.assign(state, {
    metric,
    runA,
    runB,
    rows: data.rows,
    summary: data.summary,
    expanded: null,
    detailCache: {},
  });

  $("#setup").style.display = "none";
  $("#compare-app").classList.add("active");
  $("#cmp-heading").textContent = `${metric} — A vs B`;
  $("#th-a").textContent = "A";
  $("#th-b").textContent = "B";
  $("#th-a").title = runA;
  $("#th-b").title = runB;

  renderHeader();
  renderTable();
}

function renderHeader() {
  const s = state.summary;
  $("#cmp-legend").innerHTML = `
    <div class="cmp-run"><span class="cmp-key">A</span> ${escapeHtml(state.runA)}</div>
    <div class="cmp-run"><span class="cmp-key">B</span> ${escapeHtml(state.runB)}</div>`;

  const delta = s.mean_delta;
  const tiles = [
    { label: "Run A mean", value: fmt(s.mean_a, 3) },
    { label: "Run B mean", value: fmt(s.mean_b, 3) },
    {
      label: "Mean change",
      value: signed(delta),
      tone: delta < 0 ? "bad" : delta > 0 ? "good" : "flat",
    },
    { label: "Regressed", value: `${s.n_regressed}/${s.n_paired}`, tone: "bad" },
    { label: "Improved", value: `${s.n_improved}/${s.n_paired}`, tone: "good" },
    { label: "No change", value: `${s.n_tied}/${s.n_paired}`, tone: "flat" },
  ];
  $("#cmp-tiles").innerHTML = tiles
    .map(
      (t) => `<div class="cmp-tile">
        <div class="cmp-tile-label">${t.label}</div>
        <div class="cmp-tile-value ${t.tone ? "tone-" + t.tone : ""}">${t.value}</div>
      </div>`
    )
    .join("");

  const counts = {
    all: s.n,
    regressed: s.n_regressed,
    improved: s.n_improved,
    tied: s.n_tied,
  };
  $("#cmp-filters").innerHTML = ["all", "regressed", "improved", "tied"]
    .map(
      (k) =>
        `<button class="cmp-chip ${state.filter === k ? "on" : ""}" data-filter="${k}">
          ${k === "all" ? "All" : STATUS_LABEL[k]} <span class="cmp-count">${counts[k]}</span>
        </button>`
    )
    .join("");
  for (const btn of document.querySelectorAll(".cmp-chip")) {
    btn.addEventListener("click", () => {
      state.filter = btn.dataset.filter;
      renderHeader();
      renderTable();
    });
  }
}

function sortedRows() {
  const rows = state.rows.filter(
    (r) => state.filter === "all" || r.status === state.filter
  );
  const byDelta = (a, b) => (a.delta ?? 0) - (b.delta ?? 0);
  const sorters = {
    delta: byDelta,
    "delta-desc": (a, b) => byDelta(b, a),
    name: (a, b) => a.image_path.localeCompare(b.image_path),
    "score-b": (a, b) => (a.b.score ?? 1) - (b.b.score ?? 1),
  };
  return rows.slice().sort(sorters[state.sort]);
}

// Diverging bar around a zero centre line: one hue each side, neutral middle.
function deltaBar(delta) {
  if (delta === null || delta === undefined) return "";
  const pct = Math.min(Math.abs(delta), 1) * 50;
  const side = delta < 0 ? "left" : "right";
  return `<span class="delta-bar" aria-hidden="true">
      <span class="delta-zero"></span>
      <span class="delta-fill ${delta < 0 ? "neg" : "pos"}"
            style="${side}:50%; width:${pct}%"></span>
    </span>`;
}

function renderTable() {
  const rows = sortedRows();
  $("#cmp-empty").style.display = rows.length ? "none" : "";
  $("#cmp-body").innerHTML = rows
    .map((r) => {
      const name = r.image_path.split("/").pop();
      return `<tr class="cmp-row status-${r.status}" data-image="${escapeHtml(r.image_path)}">
        <td class="col-img"><img loading="lazy" src="/api/image?path=${encodeURIComponent(
          r.image_path
        )}" alt="" /></td>
        <td class="col-name">
          <div class="cmp-name">${escapeHtml(name)}</div>
          <div class="cmp-sub">${r.a.n_atoms ?? "—"} → ${r.b.n_atoms ?? "—"} atoms
            · ${r.a.words ?? "—"} → ${r.b.words ?? "—"} words</div>
        </td>
        <td class="col-num">${fmt(r.a.score)}</td>
        <td class="col-num">${fmt(r.b.score)}</td>
        <td class="col-delta">
          <span class="delta-num ${r.delta < 0 ? "neg" : r.delta > 0 ? "pos" : ""}">${signed(
        r.delta
      )}</span>
          ${deltaBar(r.delta)}
        </td>
        <td class="col-status">
          <span class="status-light ${r.status}" aria-hidden="true"></span>
          <span class="status-word">${STATUS_LABEL[r.status]}</span>
        </td>
      </tr>
      <tr class="cmp-detail-row" data-detail="${escapeHtml(r.image_path)}"><td colspan="6">
        <div class="cmp-detail"></div>
      </td></tr>`;
    })
    .join("");

  for (const tr of document.querySelectorAll(".cmp-row")) {
    tr.addEventListener("click", () => toggleDetail(tr.dataset.image));
  }
}

// --- per-image detail ----------------------------------------------------

async function fetchDetail(run, imagePath) {
  const key = `${run}::${imagePath}`;
  if (!state.detailCache[key]) {
    const res = await fetch(
      `/api/${state.metric}/llm_judgement?run=${encodeURIComponent(
        run
      )}&image_path=${encodeURIComponent(imagePath)}`
    );
    state.detailCache[key] = res.ok ? await res.json() : null;
  }
  return state.detailCache[key];
}

// The judge stops at the first check that fails, so the first false verdict
// is the reason the atom was rejected.
const CHECK_TITLES = {
  evidence_presence_check: "the evidence named is not in the image",
  direct_check: "the vibe is not supported by the image",
  mapping_check: "the evidence is not what produces the vibe",
};

function rejectionReason(atom) {
  for (const key of ["evidence_presence_check", "direct_check", "mapping_check"]) {
    const check = atom[key];
    if (check && !check.verdict) {
      const against = check.contradicting_evidence?.length
        ? `<ul class="atom-cues">${check.contradicting_evidence
            .map((cue) => `<li>− ${escapeHtml(cue)}</li>`)
            .join("")}</ul>`
        : "";
      return `<div class="atom-why">
          <div class="atom-why-title">${CHECK_TITLES[key]}</div>
          <div class="atom-why-body">${escapeHtml(check.reasoning || "")}</div>
          ${against}
        </div>`;
    }
  }
  return "";
}

function atomList(detail) {
  const atoms = detail?.atoms || [];
  if (!atoms.length) return `<p class="cmp-muted">No atoms recorded.</p>`;
  return `<ul class="cmp-atoms">${atoms
    .map((a) => {
      if (typeof a === "string") return `<li>${escapeHtml(a)}</li>`;
      const ok = a.final_verdict;
      return `<li class="${ok ? "ok" : "fail"}">
        <span class="atom-mark">${ok ? "✓" : "✗"}</span>
        <span>${escapeHtml(a.atom)}${ok ? "" : rejectionReason(a)}</span>
      </li>`;
    })
    .join("")}</ul>`;
}

async function toggleDetail(imagePath) {
  const row = document.querySelector(`.cmp-detail-row[data-detail="${CSS.escape(imagePath)}"]`);
  if (!row) return;
  if (state.expanded === imagePath) {
    row.classList.remove("open");
    state.expanded = null;
    return;
  }
  for (const open of document.querySelectorAll(".cmp-detail-row.open")) {
    open.classList.remove("open");
  }
  state.expanded = imagePath;
  row.classList.add("open");
  const box = row.querySelector(".cmp-detail");
  box.innerHTML = `<p class="cmp-muted">Loading…</p>`;

  const [a, b] = await Promise.all([
    fetchDetail(state.runA, imagePath),
    fetchDetail(state.runB, imagePath),
  ]);
  box.innerHTML = `
    <div class="cmp-detail-grid">
      <figure class="cmp-figure">
        <img src="/api/image?path=${encodeURIComponent(imagePath)}" alt="source image" />
      </figure>
      <div class="cmp-side">
        <h3><span class="cmp-key">A</span> ${fmt(a?.score)}</h3>
        <p class="cmp-repr">${escapeHtml(a?.representation || "—")}</p>
        ${atomList(a)}
      </div>
      <div class="cmp-side">
        <h3><span class="cmp-key">B</span> ${fmt(b?.score)}</h3>
        <p class="cmp-repr">${escapeHtml(b?.representation || "—")}</p>
        ${atomList(b)}
      </div>
    </div>`;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

initSetup();
