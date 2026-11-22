"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
let state = null;
let authMode = "login";
let pollTimer = null;
let skillDirty = false;

async function api(path, opts = {}) {
  const init = { credentials: "same-origin", ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.json);
  }
  const r = await fetch(path, init);
  if (r.status === 401 && !path.startsWith("/api/auth")) { showAuth(); throw new Error("Not signed in"); }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    let msg = data.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join("; ");
    throw new Error(msg || `Request failed (${r.status})`);
  }
  return data;
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const when = (iso) => iso ? new Date(iso + (iso.endsWith("Z") ? "" : "Z")).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "—";
const flash = (el, text = "Saved") => { el.textContent = text; setTimeout(() => (el.textContent = ""), 2500); };

/* ---------------- auth ---------------- */
function showAuth() {
  $("#app").classList.add("hidden");
  $("#auth").classList.remove("hidden");
  clearInterval(pollTimer);
}
$$("[data-auth]").forEach((b) => b.addEventListener("click", () => {
  authMode = b.dataset.auth;
  $$("[data-auth]").forEach((x) => x.classList.toggle("active", x === b));
  $(".signup-only").classList.toggle("hidden", authMode !== "signup");
  $("#auth-submit").textContent = authMode === "signup" ? "Create account" : "Sign in";
  $("#auth-form [name=password]").autocomplete = authMode === "signup" ? "new-password" : "current-password";
  $("#auth-error").textContent = "";
}));
$("#auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const body = { email: f.get("email"), password: f.get("password") };
  if (authMode === "signup") body.name = f.get("name");
  $("#auth-submit").disabled = true;
  try {
    await api(`/api/auth/${authMode}`, { json: body });
    await boot();
  } catch (err) { $("#auth-error").textContent = err.message; }
  finally { $("#auth-submit").disabled = false; }
});
$("#logout").addEventListener("click", async () => { await api("/api/auth/logout", { method: "POST" }); showAuth(); });

/* ---------------- navigation ---------------- */
function showView(name) {
  $$(".nav").forEach((n) => n.classList.toggle("active", n.dataset.view === name));
  $$(".view").forEach((v) => v.classList.toggle("hidden", v.id !== `view-${name}`));
  if (name === "applications") loadApplications();
  if (name === "questions") loadQuestions();
}
$$(".nav").forEach((n) => n.addEventListener("click", () => showView(n.dataset.view)));

/* ---------------- render ---------------- */
const RUN_LABEL = { apply: "Applying", dry_run: "Dry run", profile_refresh: "Refreshing profile", login: "Signing in to Naukri", manual_login: "Waiting for browser sign-in" };
const NAUKRI_PILL = { connected: ["Connected", "ok"], needs_attention: ["Needs attention", "warn"], error: ["Error", "bad"], not_connected: ["Not connected", ""] };

function render() {
  const s = state;
  $("#user-name").textContent = s.user.name || s.user.email;

  // setup checklist
  const hasResume = s.resume && s.resume.status === "ready";
  const connected = s.naukri.status === "connected";
  $("#step-resume").classList.toggle("done", !!hasResume);
  $("#step-naukri").classList.toggle("done", connected);
  $("#step-auto").classList.toggle("done", s.settings.auto_apply);
  $("#setup").classList.toggle("hidden", hasResume && connected && s.settings.auto_apply);

  // stats
  const by = s.stats.by_status || {};
  $("#s-today").textContent = s.stats.applied_today;
  $("#s-limit").textContent = `of ${s.settings.daily_limit} daily limit`;
  $("#s-total").textContent = by.applied || 0;
  $("#s-skipped").textContent = by.skipped || 0;
  const today = new Date().toISOString().slice(0, 10);
  $("#s-refresh").textContent = s.naukri.last_profile_refresh ? (s.naukri.last_profile_refresh === today ? "Today ✓" : s.naukri.last_profile_refresh) : "Not yet";
  $("#s-refresh-note").textContent = s.naukri.last_profile_refresh_note || "";

  // settings (don't clobber while user is typing)
  if (!document.activeElement?.closest?.(".form-grid")) {
    $("#auto-apply").checked = s.settings.auto_apply;
    $("#set-limit").value = s.settings.daily_limit;
    $("#set-match").value = s.settings.min_match;
    $("#set-hour").value = s.settings.run_hour;
    $("#set-refresh").checked = s.settings.daily_profile_refresh;
    $("#set-keywords").value = s.settings.keywords;
    $("#set-locations").value = s.settings.locations;
  }
  $("#auto-desc").textContent = s.settings.auto_apply
    ? `On. Every day from ${String(s.settings.run_hour).padStart(2, "0")}:00 we ${s.settings.daily_profile_refresh ? "refresh your Naukri profile, then " : ""}apply to jobs scoring ${s.settings.min_match}%+ (up to ${s.settings.daily_limit}/day).`
    : "Off. Turn it on and every day from the start hour we refresh your Naukri profile, then apply to matching jobs up to your daily limit.";

  // run panel
  const a = s.active_run;
  $$("[data-run]").forEach((b) => (b.disabled = !!a || !connected || (b.dataset.run !== "profile_refresh" && !hasResume)));
  $("#stop-run").classList.toggle("hidden", !a);
  const pill = $("#run-status");
  if (a) { pill.textContent = `${RUN_LABEL[a.kind] || a.kind}…`; pill.className = "pill run"; $("#run-log").textContent = a.log.join("\n") || "Starting…"; }
  else if (s.last_run) {
    const r = s.last_run;
    pill.textContent = `Last: ${RUN_LABEL[r.kind] || r.kind} · ${r.status}`;
    pill.className = "pill " + (r.status === "done" ? "ok" : r.status === "error" ? "bad" : "");
    $("#run-log").textContent = `${when(r.started_at)} (${r.trigger})\n${r.summary || ""}`;
  } else { pill.textContent = ""; pill.className = "pill"; }

  // resume
  const rs = $("#resume-status");
  if (!s.resume) { rs.textContent = "No resume"; rs.className = "pill"; $("#resume-name").textContent = "Drop your resume here"; }
  else {
    $("#resume-name").textContent = s.resume.filename;
    rs.textContent = { processing: "Building skill…", ready: "Skill ready", error: "Failed" }[s.resume.status];
    rs.className = "pill " + { processing: "run", ready: "ok", error: "bad" }[s.resume.status];
    $("#resume-error").textContent = s.resume.status === "error" ? s.resume.error : "";
  }
  $("#skill-card").classList.toggle("hidden", !hasResume);
  if (hasResume && !skillDirty) fillSkill(s.resume.skill);

  // naukri
  const [label, cls] = NAUKRI_PILL[s.naukri.status] || [s.naukri.status, ""];
  $("#naukri-status").textContent = a && ["login", "manual_login"].includes(a.kind) ? RUN_LABEL[a.kind] + "…" : label;
  $("#naukri-status").className = "pill " + (a && ["login", "manual_login"].includes(a.kind) ? "run" : cls);
  $("#naukri-msg").textContent = s.naukri.message || "";
  if (document.activeElement !== $("#nk-email")) $("#nk-email").value = s.naukri.email || "";
  $("#nk-pass-hint").textContent = s.naukri.has_password ? "(saved — leave empty to keep)" : "";

  // questions badge
  const q = s.stats.open_questions;
  $("#q-badge").textContent = q; $("#q-badge").classList.toggle("hidden", !q);
}

/* ---------------- skill ---------------- */
const LIST_FIELDS = ["search_keywords", "target_titles", "core_skills", "secondary_skills", "avoid_keywords", "preferred_locations"];
const ANSWER_FIELDS = ["notice_period", "current_ctc", "expected_ctc", "willing_to_relocate"];
function fillSkill(sk) {
  $("#sk-headline").value = sk.headline || "";
  LIST_FIELDS.forEach((k) => ($(`#sk-${k}`).value = (sk[k] || []).join("\n")));
  $("#sk-experience_years").value = sk.experience_years ?? "";
  $("#sk-seniority").value = sk.seniority || "";
  ANSWER_FIELDS.forEach((k) => ($(`#sk-${k}`).value = (sk.standard_answers || {})[k] || ""));
}
$("#skill-card").addEventListener("input", () => (skillDirty = true));
$("#save-skill").addEventListener("click", async () => {
  const skill = { headline: $("#sk-headline").value.trim(), seniority: $("#sk-seniority").value.trim(),
    experience_years: parseFloat($("#sk-experience_years").value) || 0, standard_answers: {} };
  LIST_FIELDS.forEach((k) => (skill[k] = $(`#sk-${k}`).value.split(/\n|,/).map((x) => x.trim()).filter(Boolean)));
  ANSWER_FIELDS.forEach((k) => (skill.standard_answers[k] = $(`#sk-${k}`).value.trim()));
  try { await api("/api/skill", { method: "PUT", json: { skill } }); skillDirty = false; flash($("#skill-saved")); refresh(); }
  catch (e) { flash($("#skill-saved"), e.message); }
});
$("#rebuild-skill").addEventListener("click", async () => {
  if (!confirm("Rebuild the skill from your resume? Your edits to the skill will be replaced.")) return;
  skillDirty = false; await api("/api/skill/rebuild", { method: "POST" }); refresh();
});

/* ---------------- resume upload ---------------- */
async function upload(file) {
  $("#resume-error").textContent = "";
  const fd = new FormData(); fd.append("file", file);
  try { skillDirty = false; await api("/api/resume", { method: "POST", body: fd }); refresh(); }
  catch (e) { $("#resume-error").textContent = e.message; }
}
$("#pick-file").addEventListener("click", () => $("#resume-file").click());
$("#resume-file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));

/* ---------------- settings & runs ---------------- */
async function saveSettings(extra = {}) {
  await api("/api/settings", { method: "PUT", json: {
    daily_limit: parseInt($("#set-limit").value) || 25, min_match: parseInt($("#set-match").value) || 60,
    run_hour: parseInt($("#set-hour").value) || 0, daily_profile_refresh: $("#set-refresh").checked,
    keywords: $("#set-keywords").value.trim(), locations: $("#set-locations").value.trim(), ...extra } });
}
$("#save-settings").addEventListener("click", async () => {
  try { await saveSettings(); flash($("#settings-saved")); refresh(); } catch (e) { flash($("#settings-saved"), e.message); }
});
$("#auto-apply").addEventListener("change", async (e) => {
  const on = e.target.checked;
  if (on && (state.resume?.status !== "ready" || state.naukri.status !== "connected")) {
    e.target.checked = false; alert("Upload your resume and connect Naukri first."); return;
  }
  try { await saveSettings({ auto_apply: on }); refresh(); } catch (err) { e.target.checked = !on; alert(err.message); }
});
$$("[data-run]").forEach((b) => b.addEventListener("click", async () => {
  $("#run-error").textContent = "";
  try { await api("/api/runs", { json: { kind: b.dataset.run } }); refresh(); }
  catch (e) { $("#run-error").textContent = e.message; }
}));
$("#stop-run").addEventListener("click", async () => { await api("/api/runs/stop", { method: "POST" }); refresh(); });

/* ---------------- naukri ---------------- */
$("#naukri-form").addEventListener("submit", async (e) => {
  e.preventDefault(); $("#naukri-error").textContent = "";
  try { await api("/api/naukri/connect", { json: { email: $("#nk-email").value.trim(), password: $("#nk-password").value } });
    $("#nk-password").value = ""; refresh(); }
  catch (err) { $("#naukri-error").textContent = err.message; }
});
$("#browser-login").addEventListener("click", async () => {
  $("#naukri-error").textContent = "";
  try { await api("/api/naukri/browser-login", { method: "POST" }); $("#naukri-msg").textContent = "A browser window opened on the server machine — sign in to Naukri there (10 minutes)."; refresh(); }
  catch (err) { $("#naukri-error").textContent = err.message; }
});
$("#naukri-check").addEventListener("click", async () => {
  $("#naukri-error").textContent = "";
  try { await api("/api/naukri/check", { method: "POST" }); refresh(); }
  catch (err) { $("#naukri-error").textContent = err.message; }
});
$("#naukri-disconnect").addEventListener("click", async () => {
  if (!confirm("Disconnect Naukri? Auto-apply will be turned off.")) return;
  await api("/api/naukri", { method: "DELETE" }); refresh();
});

/* ---------------- applications ---------------- */
const STATUS_PILL = { applied: "ok", skipped: "", external: "warn", needs_user: "warn", error: "bad", unconfirmed: "bad", dry_run: "run" };
const STATUS_LABEL = { external: "company site", needs_user: "needs you", dry_run: "dry run" };
async function loadApplications() {
  const rows = await api(`/api/applications?status=${$("#app-filter").value}`);
  $("#app-rows").innerHTML = rows.length ? rows.map((a) => `<tr>
    <td>${esc(when(a.created_at))}</td>
    <td><a href="${esc(a.url)}" target="_blank" rel="noopener">${esc(a.title || "(untitled)")}</a></td>
    <td>${esc(a.company)}</td><td>${a.score || "—"}</td>
    <td><span class="pill ${STATUS_PILL[a.status] || ""}">${esc(STATUS_LABEL[a.status] || a.status)}</span></td>
    <td class="note">${esc(a.message)}</td></tr>`).join("") : `<tr><td colspan="6" class="empty">No applications yet.</td></tr>`;
}
$("#app-filter").addEventListener("change", loadApplications);

/* ---------------- questions ---------------- */
async function loadQuestions() {
  const qs = await api("/api/questions");
  $("#q-list").innerHTML = qs.length ? qs.map((q) => `<div class="q ${q.answer ? "" : "open"}">
    <p>${esc(q.question)}</p>${q.options.length ? `<div class="opts">Options: ${q.options.map(esc).join(" · ")}</div>` : ""}
    <form class="row" data-q="${q.id}">
      ${q.options.length ? `<select class="grow"><option value="">Choose…</option>${q.options.map((o) => `<option ${o === q.answer ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`
        : `<input class="grow" value="${esc(q.answer)}" placeholder="Your answer">`}
      <button class="btn">${q.answer ? "Update" : "Save"}</button></form></div>`).join("")
    : `<div class="empty">No questions yet. They show up here when an employer asks something your resume doesn't cover.</div>`;
  $$("#q-list form").forEach((f) => f.addEventListener("submit", async (e) => {
    e.preventDefault();
    const val = f.querySelector("input, select").value.trim(); if (!val) return;
    await api(`/api/questions/${f.dataset.q}`, { method: "PUT", json: { answer: val } });
    loadQuestions(); refresh();
  }));
}
$("#q-add").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#q-new").value.trim(), answer = $("#q-new-answer").value.trim();
  if (!question || !answer) return;
  await api("/api/questions", { json: { question, answer } });
  $("#q-new").value = $("#q-new-answer").value = ""; loadQuestions();
});

/* ---------------- boot ---------------- */
async function refresh() {
  try { state = await api("/api/me"); render(); } catch { /* handled */ }
}
async function boot() {
  try { state = await api("/api/me"); } catch { showAuth(); return; }
  $("#auth").classList.add("hidden"); $("#app").classList.remove("hidden");
  render();
  if (!state.resume) showView("skill");
  clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    const busy = state?.active_run || state?.resume?.status === "processing";
    if (busy || Date.now() % 30000 < 4000) refresh();
  }, 4000);
}
boot();
