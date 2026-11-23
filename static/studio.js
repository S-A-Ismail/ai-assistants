"use strict";

const CATEGORIES = [
  ["image", "Image"], ["edit", "Edit"], ["animate", "Animate"], ["voice", "Voice-over"],
  ["music", "Music"], ["translate", "Translate"], ["transcribe", "Transcribe"],
];
const FIELD_TYPES_WITH_FILE = new Set(["image", "audio"]);

const state = {
  tools: [],
  tab: localStorage.getItem("tab") || "image",
  tool: null,
  refs: {},        // field name -> "job:<id>/<file>" chosen via "Use in…"
  pendingRef: null, // { ref, kind } carried across a tab switch
  jobs: [],
};

const $ = (s, el = document) => el.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids) if (kid != null) e.append(kid);
  return e;
};

// ---------------------------------------------------------------- API (with optional key)

function key() { try { return localStorage.getItem("apiKey") || ""; } catch { return ""; } }

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (key()) headers["X-API-Key"] = key();
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  if (r.status === 401) {
    const k = prompt("This studio needs its API key (STUDIO_API_KEY):");
    if (k) { localStorage.setItem("apiKey", k); return api(path, opts); }
  }
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

function fileUrl(url) { return key() ? `${url}?key=${encodeURIComponent(key())}` : url; }

// ---------------------------------------------------------------- tabs and tools

function renderTabs() {
  const nav = $("#tabs");
  nav.replaceChildren(...CATEGORIES.map(([id, label]) => {
    const n = state.tools.filter(t => t.category === id);
    const ready = n.filter(t => t.ready).length;
    return el("button", {
      role: "tab", "aria-selected": state.tab === id ? "true" : "false",
      onclick: () => selectTab(id),
    }, `${label} `, el("span", { class: "chip", text: `${ready}/${n.length}` }));
  }));
}

function selectTab(id) {
  state.tab = id;
  try { localStorage.setItem("tab", id); } catch {}
  const tools = state.tools.filter(t => t.category === id);
  const remembered = localStorage.getItem(`tool:${id}`);
  state.tool = tools.find(t => t.id === remembered) || tools.find(t => t.ready) || tools[0];
  renderTabs();
  renderTools();
  renderForm();
  renderJobs();
}

function renderTools() {
  const tools = state.tools.filter(t => t.category === state.tab);
  $("#tools").replaceChildren(...tools.map(t => el("button", {
    class: "tool", type: "button", "aria-pressed": state.tool && t.id === state.tool.id ? "true" : "false",
    onclick: () => { state.tool = t; localStorage.setItem(`tool:${state.tab}`, t.id); renderTools(); renderForm(); },
  },
    el("div", { class: "t1" }, el("span", { text: t.title }),
      el("span", {}, t.commercial ? null : el("span", { class: "chip nc", text: "non-commercial" }), " ",
        el("span", { class: t.ready ? "chip" : "chip off", text: t.ready ? t.speed : "not ready" }))),
    el("div", { class: "t2", text: t.summary }),
  )));
}

// ---------------------------------------------------------------- form

function renderForm() {
  const t = state.tool;
  state.refs = {};
  if (!t) { $("#form").hidden = true; return; }
  $("#form").hidden = false;
  $("#toolhead").replaceChildren(
    el("h3", { text: t.title }),
    el("div", { class: "meta" }, `${t.model} · ${t.license}`, t.ready ? null : el("div", { class: "msg err", text: t.status })),
  );
  const box = $("#fields");
  box.replaceChildren();
  const numbers = [];
  for (const f of t.fields) {
    const node = fieldNode(f);
    if (f.type === "number" || f.type === "bool") numbers.push(node);
    else {
      flush();
      box.append(node);
    }
  }
  flush();
  function flush() {
    while (numbers.length) box.append(el("div", { class: "row2" }, ...numbers.splice(0, 2)));
  }
  $("#go").disabled = !t.ready;
  $("#formmsg").textContent = "";
  $("#formmsg").className = "msg";

  // A result chosen with "Use in…" on another tab lands in the first matching file field.
  if (state.pendingRef) {
    const f = t.fields.find(f => f.type === state.pendingRef.kind);
    if (f) setRef(f.name, state.pendingRef.ref, state.pendingRef.url);
    state.pendingRef = null;
  }
}

function fieldNode(f) {
  const id = `f_${f.name}`;
  const label = el("label", { for: id, text: f.label + (f.required ? " *" : "") });
  let input;
  if (f.type === "textarea") {
    input = el("textarea", { id, name: f.name, required: f.required });
    input.value = f.default ?? "";
  } else if (f.type === "select") {
    input = el("select", { id, name: f.name });
    for (const o of f.options || []) input.append(el("option", { value: o, text: o === "" ? "—" : o }));
    if (f.default != null) input.value = f.default;
  } else if (f.type === "bool") {
    input = el("input", { id, name: f.name, type: "checkbox" });
    input.checked = !!f.default;
    return el("div", { class: "field" }, el("label", { for: id }, input, " ", f.label));
  } else if (f.type === "number") {
    input = el("input", { id, name: f.name, type: "number", min: f.min, max: f.max, step: f.step ?? "any" });
    input.value = f.default ?? "";
  } else if (FIELD_TYPES_WITH_FILE.has(f.type)) {
    const accept = f.type === "image" ? "image/*" : "audio/*,video/*";
    input = el("input", { id, name: f.name, type: "file", accept, onchange: e => previewFile(f, e.target) });
    const drop = el("div", { class: "drop", id: `drop_${f.name}` }, input);
    return el("div", { class: "field" }, label, drop, f.help ? el("div", { class: "help", text: f.help }) : null);
  } else {
    input = el("input", { id, name: f.name, type: "text" });
    input.value = f.default ?? "";
  }
  return el("div", { class: "field" }, label, input, f.help ? el("div", { class: "help", text: f.help }) : null);
}

function previewFile(f, input) {
  delete state.refs[f.name];
  const drop = $(`#drop_${f.name}`);
  drop.querySelectorAll("img,.ref,.clear").forEach(n => n.remove());
  const file = input.files[0];
  if (file && f.type === "image") drop.prepend(el("img", { src: URL.createObjectURL(file), alt: "" }));
}

function setRef(name, ref, url) {
  state.refs[name] = ref;
  const drop = $(`#drop_${name}`);
  if (!drop) return;
  drop.querySelectorAll("img,.ref,.clear").forEach(n => n.remove());
  const input = drop.querySelector("input");
  input.value = "";
  if (input.accept.startsWith("image")) drop.prepend(el("img", { src: fileUrl(url), alt: "" }));
  drop.append(el("span", { class: "ref", text: `using ${ref}` }),
    el("button", { type: "button", class: "small clear", text: "clear", onclick: () => { delete state.refs[name]; setRef0(drop); } }));
}
function setRef0(drop) { drop.querySelectorAll("img,.ref,.clear").forEach(n => n.remove()); }

$("#form").addEventListener("submit", async e => {
  e.preventDefault();
  const t = state.tool;
  const fd = new FormData();
  fd.append("tool", t.id);
  for (const f of t.fields) {
    const input = document.getElementById(`f_${f.name}`);
    if (FIELD_TYPES_WITH_FILE.has(f.type)) {
      if (state.refs[f.name]) fd.append(f.name, state.refs[f.name]);
      else if (input.files[0]) fd.append(f.name, input.files[0]);
    } else if (f.type === "bool") fd.append(f.name, input.checked ? "true" : "false");
    else fd.append(f.name, input.value);
  }
  const msg = $("#formmsg");
  $("#go").disabled = true;
  msg.className = "msg";
  msg.textContent = "uploading…";
  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    msg.textContent = `queued (${job.id})`;
    await refreshJobs();
  } catch (err) {
    msg.className = "msg err";
    msg.textContent = err.message;
  } finally {
    $("#go").disabled = !t.ready;
  }
});

// ---------------------------------------------------------------- jobs

function toolOf(job) { return state.tools.find(t => t.id === job.tool) || { title: job.tool, model: "", category: "" }; }

function summary(job) {
  const p = job.params || {};
  return p.prompt || p.text || p.caption || (p.image ? "" : "") || "";
}

function renderJobs() {
  const only = $("#onlytab").checked;
  const jobs = state.jobs.filter(j => !only || toolOf(j).category === state.tab);
  const box = $("#jobs");
  if (!jobs.length) { box.replaceChildren(el("div", { class: "empty", text: "No jobs yet. Results show up here, and stay after restarts." })); return; }
  box.replaceChildren(...jobs.map(jobNode));
}

function jobNode(job) {
  const t = toolOf(job);
  const n = $("#jobtpl").content.firstElementChild.cloneNode(true);
  const finished = ["done", "error", "cancelled"].includes(job.status);
  if (finished) n.classList.add("finished");
  $(".jtitle", n).textContent = t.title;
  $(".jmodel", n).textContent = t.model;
  const st = $(".jstatus", n);
  st.classList.add(job.status);
  const pct = Math.round((job.progress || 0) * 100);
  const time = job.elapsed != null ? ` · ${fmtTime(job.elapsed)}` : "";
  st.textContent = job.status === "running" ? `${job.message || "running"} · ${pct}%${time}`
    : job.status === "queued" ? "queued" : `${job.status}${time}`;
  $(".jprompt", n).textContent = summary(job);
  $(".bar i", n).style.width = `${job.status === "running" ? Math.max(pct, 2) : 0}%`;

  const out = $(".jout", n);
  const actions = $(".jactions", n);
  if (job.status === "error") out.append(el("div", { class: "err", text: job.error }));
  const r = job.result || {};
  if (r.text) out.append(el("pre", { text: r.text }));
  if (r.translation) out.append(el("pre", { text: r.translation }));
  for (const f of job.files || []) {
    const url = fileUrl(f.url);
    const ext = f.name.split(".").pop().toLowerCase();
    const isMain = f === job.files[0];
    if (isMain && ["png", "jpg", "jpeg", "webp"].includes(ext)) out.append(el("a", { href: url, target: "_blank" }, el("img", { src: url, alt: f.name, loading: "lazy" })));
    else if (isMain && ext === "mp4") out.append(el("video", { src: url, controls: true, loop: true, muted: true, playsinline: true }));
    else if (isMain && ["mp3", "wav"].includes(ext)) out.append(el("audio", { src: url, controls: true }));
    actions.append(el("a", { class: "small", href: `${url}${url.includes("?") ? "&" : "?"}download=1`, text: `⤓ ${f.name}` }));
    if (["png", "jpg", "jpeg", "webp"].includes(ext)) {
      actions.append(useIn(job, f, "image", [["edit", "Edit"], ["animate", "Animate"]]));
    }
  }
  if (r.seed != null) actions.append(el("span", { class: "chip", text: `seed ${r.seed}` }));
  if (r.hindi_remapped) actions.append(el("span", { class: "chip", text: "Hindi→Urdu" }));
  if (!finished) actions.append(el("button", { class: "small", text: "Cancel", onclick: () => act(`/api/jobs/${job.id}/cancel`, "POST") }));
  else actions.append(el("button", { class: "small", text: "Delete", onclick: () => confirm("Delete this job and its files?") && act(`/api/jobs/${job.id}`, "DELETE") }));
  return n;
}

function useIn(job, file, kind, targets) {
  const sel = el("select", { class: "small", "aria-label": "Use this result in another tool" },
    el("option", { value: "", text: "Use in…" }),
    ...targets.map(([id, label]) => el("option", { value: id, text: label })));
  sel.addEventListener("change", () => {
    if (!sel.value) return;
    state.pendingRef = { ref: `job:${job.id}/${file.name}`, url: file.url, kind };
    selectTab(sel.value);
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  return sel;
}

async function act(path, method) {
  try { await api(path, { method }); } catch (e) { alert(e.message); }
  refreshJobs();
}

function fmtTime(s) {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

// ---------------------------------------------------------------- polling

async function refreshJobs() {
  try { state.jobs = await api("/api/jobs?limit=60"); renderJobs(); } catch {}
}

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    const q = h.queue.running.length ? `running 1, queued ${h.queue.queued}` : h.queue.queued ? `queued ${h.queue.queued}` : "idle";
    const models = h.models_present === h.models_total ? "all models present"
      : `models ${h.models_present}/${h.models_total} (${h.models_missing_gb} GB still to download)`;
    $("#health").replaceChildren(el("b", { text: `${h.tools_ready}/${h.tools_total} tools ready` }), ` · ${models} · ${q}`);
  } catch (e) { $("#health").textContent = `server unreachable: ${e.message}`; }
}

let timer;
function schedule() {
  clearTimeout(timer);
  const busy = state.jobs.some(j => j.status === "running" || j.status === "queued");
  timer = setTimeout(async () => { await refreshJobs(); await refreshHealth(); schedule(); }, busy ? 3000 : 15000);
}

$("#onlytab").addEventListener("change", renderJobs);

(async function init() {
  try {
    state.tools = await api("/api/tools");
  } catch (e) {
    $("#health").textContent = `server unreachable: ${e.message}`;
    return;
  }
  selectTab(CATEGORIES.some(([id]) => id === state.tab) ? state.tab : "image");
  await Promise.all([refreshJobs(), refreshHealth()]);
  schedule();
  // Tool readiness changes as downloads finish; re-check now and then.
  setInterval(async () => {
    try {
      const fresh = await api("/api/tools");
      if (!fresh.some(t => (state.tools.find(o => o.id === t.id) || {}).ready !== t.ready)) return;
      const wasReady = state.tool && state.tool.ready;
      state.tools = fresh;
      state.tool = state.tool && fresh.find(t => t.id === state.tool.id);
      renderTabs();
      renderTools();
      // Only rebuild the form (losing typed input) if the selected tool itself changed state.
      if (state.tool && state.tool.ready !== wasReady) renderForm();
    } catch {}
  }, 60000);
})();
