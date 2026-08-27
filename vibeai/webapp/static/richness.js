const state = {
  run: null,
  annotator: null,
  items: [],
  annotations: {},
  index: 0,
  draft: null,
  savedSnapshot: null,
  axes: [],              // the scored axes, in canonical order
  extraAxes: [],         // annotatable but unscored ("other")
  alwaysApplicable: [],  // axes the annotator can't mark N/A
  axisInfo: {},          // slug -> {label, description, applicability}
  droppedLegacyTags: 0,  // axis tags on the loaded item from an older vocabulary
};

const $ = (sel) => document.querySelector(sel);

function axisLabel(axis) {
  return state.axisInfo[axis]?.label || axis.replace(/-/g, " ");
}

function axisTitle(axis) {
  const info = state.axisInfo[axis];
  if (!info) return "";
  return `${info.description}\n${info.applicability}`;
}

function isLocked(axis) {
  return state.alwaysApplicable.includes(axis);
}

function draftKey(d) {
  return JSON.stringify(d);
}

function isDirty() {
  if (!state.draft) return false;
  return draftKey(state.draft) !== state.savedSnapshot;
}

// Atoms arrive either as plain strings (decomposition_quality runs) or as
// objects carrying the judge's atom/type/stated_* fields (plausibility runs).
function normalizeAtom(a) {
  if (typeof a === "string") return { atom: a };
  return a;
}

function blankDraft(item) {
  return {
    // Everything is applicable until the annotator says otherwise — the same
    // default the judge prompt uses, so N/A is always a deliberate call.
    applicable: [...state.axes],
    atoms: item.atoms.map(normalizeAtom).map((a) => ({
      atom: a.atom,
      type: a.type || null,
      stated_vibe: a.stated_vibe || null,
      stated_evidence: a.stated_evidence || null,
      axes: [],
      reason: "",
    })),
  };
}

function draftFromAnnotation(item, annotation) {
  const blank = blankDraft(item);
  state.droppedLegacyTags = 0;
  if (!annotation) return blank;
  const known = new Set([...state.axes, ...state.extraAxes]);
  return {
    // Records saved before per-image applicability existed have no
    // applicable_axes: they were scored against the whole axis set.
    applicable: annotation.applicable_axes
      ? annotation.applicable_axes.filter((a) => state.axes.includes(a))
      : [...state.axes],
    atoms: blank.atoms.map((entry, idx) => {
      const saved = annotation.atoms[idx];
      if (!saved) return entry;
      const kept = saved.axes.filter((a) => known.has(a));
      state.droppedLegacyTags += saved.axes.length - kept.length;
      return { ...entry, axes: kept, reason: saved.reason || "" };
    }),
  };
}

// --- setup screen --------------------------------------------------------

async function initSetup() {
  const savedAnnotator = localStorage.getItem("rich_annotator") || "";
  const savedRun = localStorage.getItem("rich_run") || "";
  $("#annotator-input").value = savedAnnotator;

  const [axesRes, runsRes] = await Promise.all([
    fetch("/api/richness/axes"),
    fetch("/api/richness/runs"),
  ]);
  const axesData = await axesRes.json();
  state.axes = axesData.axes;
  state.extraAxes = axesData.extra_axes;
  state.alwaysApplicable = axesData.always_applicable || [];
  state.axisInfo = axesData.info || {};
  renderAxisDefinitions();

  const { runs } = await runsRes.json();
  const sel = $("#run-select");
  sel.innerHTML = "";
  const groups = {};
  for (const r of runs) {
    const [sourceMetric, name] = r.includes("/") ? [r.slice(0, r.indexOf("/")), r.slice(r.indexOf("/") + 1)] : ["", r];
    if (!groups[sourceMetric]) {
      const g = document.createElement("optgroup");
      g.label = sourceMetric || "runs";
      sel.appendChild(g);
      groups[sourceMetric] = g;
    }
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = name;
    groups[sourceMetric].appendChild(opt);
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

function renderAxisDefinitions() {
  const ul = $("#axis-definitions");
  ul.innerHTML = "";
  for (const axis of [...state.axes, ...state.extraAxes]) {
    const info = state.axisInfo[axis] || {};
    const li = document.createElement("li");
    li.innerHTML =
      `<strong>${escapeHtml(axisLabel(axis))}</strong> — ${escapeHtml(info.description || "")}` +
      `<span class="axis-applicability">${escapeHtml(info.applicability || "")}</span>`;
    ul.appendChild(li);
  }
  $("#axis-total").textContent = String(state.axes.length);
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

  renderApplicability();
  renderAtoms();
  renderCoverage();
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

// --- applicability: the per-image denominator -----------------------------

function isApplicable(axis) {
  return state.draft.applicable.includes(axis);
}

function renderApplicability() {
  const row = $("#applicability-chips");
  row.innerHTML = "";
  for (const axis of state.axes) {
    const btn = document.createElement("button");
    btn.className = "applic-chip";
    btn.type = "button";
    const locked = isLocked(axis);
    if (locked) {
      btn.classList.add("locked");
      btn.disabled = true;
      btn.title = `${axisTitle(axis)}\n\nAlways applicable — can't be marked N/A.`;
    } else {
      btn.title = `${axisTitle(axis)}\n\nClick to mark this axis N/A for this image.`;
    }
    const mark = document.createElement("span");
    mark.className = "applic-chip-mark";
    btn.appendChild(mark);
    btn.appendChild(document.createTextNode(axisLabel(axis)));
    const refresh = () => {
      const on = isApplicable(axis);
      btn.classList.toggle("na", !on);
      btn.setAttribute("aria-pressed", String(on));
      mark.textContent = on ? "✓" : "–";
    };
    btn.addEventListener("click", () => {
      if (locked) return;
      state.draft.applicable = isApplicable(axis)
        ? state.draft.applicable.filter((a) => a !== axis)
        : state.axes.filter((a) => a === axis || isApplicable(a));
      refresh();
      updateApplicabilityCount();
      renderAtoms();
      renderCoverage();
      setSaveStatus("Unsaved changes");
    });
    refresh();
    row.appendChild(btn);
  }
  updateApplicabilityCount();
}

function updateApplicabilityCount() {
  const applicable = state.draft.applicable.length;
  const na = state.axes.length - applicable;
  $("#applicability-count").textContent =
    `${applicable} / ${state.axes.length} applicable` + (na ? ` — ${na} marked n/a` : "");
}

function renderAtoms() {
  const container = $("#atoms-container");
  container.innerHTML = "";
  state.draft.atoms.forEach((entry, idx) => {
    const card = document.createElement("div");
    card.className = "atom-card";

    const meta = document.createElement("div");
    const evidence = Array.isArray(entry.stated_evidence)
      ? entry.stated_evidence.map(escapeHtml).join("; ")
      : entry.stated_evidence
        ? escapeHtml(entry.stated_evidence)
        : "";
    meta.innerHTML = `
      ${entry.type ? `<div class="atom-type-badge">${escapeHtml(entry.type.replace("_", " "))}</div>` : ""}
      <div class="atom-text">${idx + 1}. ${escapeHtml(entry.atom)}</div>
      ${entry.stated_vibe || evidence ? `<div class="atom-meta">
        ${entry.stated_vibe ? `Stated vibe: <strong>${escapeHtml(entry.stated_vibe)}</strong>` : ""}
        ${evidence ? `<br/>Stated evidence: ${evidence}` : ""}
      </div>` : ""}
    `;
    card.appendChild(meta);

    const axisRow = document.createElement("div");
    axisRow.className = "axis-group";
    for (const axis of [...state.axes, ...state.extraAxes]) {
      const btn = document.createElement("button");
      btn.className = "axis-chip";
      btn.type = "button";
      const scored = state.axes.includes(axis);
      const na = scored && !isApplicable(axis);
      btn.title = na
        ? `${axisTitle(axis)}\n\nMarked N/A for this image — tagging it here won't be scored.`
        : axisTitle(axis);
      if (state.extraAxes.includes(axis)) btn.classList.add("extra");
      if (na) btn.classList.add("na");
      // Checkmark is always in the DOM (hidden when off) so toggling a chip
      // never reflows the row.
      const mark = document.createElement("span");
      mark.className = "axis-chip-mark";
      mark.textContent = "✓";
      btn.appendChild(mark);
      btn.appendChild(document.createTextNode(axisLabel(axis)));
      const refresh = () => {
        const on = entry.axes.includes(axis);
        btn.classList.toggle("selected", on);
        btn.setAttribute("aria-pressed", String(on));
      };
      btn.addEventListener("click", () => {
        if (entry.axes.includes(axis)) {
          entry.axes = entry.axes.filter((a) => a !== axis);
        } else {
          entry.axes = [...entry.axes, axis];
        }
        refresh();
        renderCoverage();
        setSaveStatus("Unsaved changes");
      });
      refresh();
      axisRow.appendChild(btn);
    }
    card.appendChild(axisRow);

    const reasonInput = document.createElement("textarea");
    reasonInput.className = "reason-input";
    reasonInput.placeholder = "Why these axes? (optional)";
    reasonInput.value = entry.reason || "";
    reasonInput.addEventListener("input", () => {
      entry.reason = reasonInput.value;
      setSaveStatus("Unsaved changes");
    });
    card.appendChild(reasonInput);

    container.appendChild(card);
  });
}

function renderCoverage() {
  const counts = {};
  for (const axis of [...state.axes, ...state.extraAxes]) {
    counts[axis] = state.draft.atoms.filter((a) => a.axes.includes(axis)).length;
  }
  const chips = $("#coverage-chips");
  chips.innerHTML = "";
  for (const axis of [...state.axes, ...state.extraAxes]) {
    const scored = state.axes.includes(axis);
    const na = scored && !isApplicable(axis);
    const chip = document.createElement("span");
    chip.className = "coverage-chip";
    if (counts[axis] && !na) chip.classList.add("covered");
    if (na) chip.classList.add("na");
    if (state.extraAxes.includes(axis)) chip.classList.add("extra");
    const label = na ? `${axisLabel(axis)} · n/a` : axisLabel(axis);
    chip.textContent = counts[axis] ? `${label} · ${counts[axis]}` : label;
    chips.appendChild(chip);
  }
  const applicable = state.draft.applicable;
  const total = applicable.length;
  const covered = applicable.filter((axis) => counts[axis]).length;
  const untagged = state.draft.atoms.filter((a) => !a.axes.length).length;
  const conflicting = state.axes.filter((axis) => !isApplicable(axis) && counts[axis]);

  $("#coverage-count").textContent = `${covered} / ${total}`;
  $("#coverage-count-label").textContent =
    total === state.axes.length
      ? "applicable axes covered"
      : `applicable axes covered (${state.axes.length - total} n/a)`;
  $("#coverage-score").textContent = `score ${(total ? covered / total : 0).toFixed(2)}`;
  $("#coverage-fill").style.width = `${total ? (covered / total) * 100 : 0}%`;

  const notes = $("#coverage-notes");
  notes.innerHTML = "";
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  if (counts.other) notes.appendChild(note(`${plural(counts.other, "atom")} tagged other — not scored`));
  if (untagged) notes.appendChild(note(`${plural(untagged, "atom")} still untagged`));
  if (conflicting.length) {
    notes.appendChild(
      note(`tagged but marked n/a: ${conflicting.map(axisLabel).join(", ")} — not scored`, "warn")
    );
  }
  if (state.droppedLegacyTags) {
    notes.appendChild(
      note(
        `${plural(state.droppedLegacyTags, "tag")} from the old 8-axis vocabulary dropped — re-tag before saving`,
        "warn"
      )
    );
  }
}

function note(text, extraClass) {
  const el = document.createElement("span");
  el.className = extraClass ? `coverage-note ${extraClass}` : "coverage-note";
  el.textContent = text;
  return el;
}

function setSaveStatus(text) {
  $("#save-status").textContent = text;
}

async function saveCurrent() {
  const item = state.items[state.index];
  const body = {
    run: state.run,
    annotator: state.annotator,
    image_path: item.image_path,
    applicable_axes: state.draft.applicable,
    atoms: state.draft.atoms.map((a) => ({
      atom: a.atom,
      axes: a.axes,
      reason: a.reason ? a.reason.trim() || null : null,
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
  state.droppedLegacyTags = 0;
  setSaveStatus(`Saved ✓ ${new Date(saved.updated_at).toLocaleTimeString()}`);
  renderSidebar();
  renderCoverage();
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

  $("#toggle-image-btn").addEventListener("click", () => {
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

  $("#toggle-rubric-btn").addEventListener("click", () => {
    const body = $("#rubric-body");
    body.classList.toggle("open");
    $("#toggle-rubric-btn").textContent = body.classList.contains("open")
      ? "Hide axis definitions"
      : "Show axis definitions";
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
