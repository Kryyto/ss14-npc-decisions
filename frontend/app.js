"use strict";

/* ---------------------------------------------------------------------------
 * Backend base URL — the only place the API origin is configured.
 * Local dev default: uvicorn on port 7860. Point this at your deployed
 * backend (e.g. the HF Space URL) for the hosted frontend.
 * ------------------------------------------------------------------------- */
const API_BASE_URL = ["localhost", "127.0.0.1"].includes(location.hostname)
  ? "http://localhost:7860"
  : "https://kryyto-ss14-npc-decisions.hf.space";

const COLD_START_DELAY_MS = 3000;
const COLD_START_GAP_MS = 5000;
const JOB_LOAD_ATTEMPTS = 30;
const JOB_LOAD_RETRY_MS = 2000;
const MESSAGE_MAX = 2000;
const LANG_STORAGE_KEY = "ss14-npc-lang";

const els = {
  form: document.getElementById("predict-form"),
  job: document.getElementById("job"),
  jobDesc: document.getElementById("job-desc"),
  message: document.getElementById("message"),
  charCount: document.getElementById("char-count"),
  submit: document.getElementById("submit"),
  status: document.getElementById("status"),
  error: document.getElementById("error"),
  results: document.getElementById("results"),
  answers: document.getElementById("answers"),
  requestId: document.getElementById("request-id"),
  langButtons: Array.from(document.querySelectorAll(".lang-btn")),
  dossier: document.getElementById("dossier"),
  charSprite: document.getElementById("char-sprite"),
  charName: document.getElementById("char-name"),
  charRole: document.getElementById("char-role"),
  charDept: document.getElementById("char-dept"),
  charEmployeeId: document.getElementById("char-employee-id"),
  charSpecies: document.getElementById("char-species"),
  charAssignment: document.getElementById("char-assignment"),
  charState: document.getElementById("char-state"),
  charLocation: document.getElementById("char-location"),
  mentalChart: document.getElementById("mental-chart"),
  mentalPoint: document.getElementById("mental-point"),
  mentalPointGlow: document.getElementById("mental-point-glow"),
  mentalGuideX: document.getElementById("mental-guide-x"),
  mentalGuideY: document.getElementById("mental-guide-y"),
  mentalXLabel: document.getElementById("mental-x-label"),
  mentalYLabel: document.getElementById("mental-y-label"),
  tirednessVal: document.getElementById("mental-tiredness-val"),
  stressVal: document.getElementById("mental-stress-val"),
};

const CHART = { x0: 38, x1: 242, y0: 18, y1: 188 };
const SPRITE_RE = /^assets\/ss14\/[a-z-]+\.png$/;
const SPRITE_BODY = "assets/ss14/human-full.png";
const SPRITE_LAYERS = {
  janitor: { hair: "assets/ss14/hair-messy.png", shoes: "assets/ss14/janitor-shoes.png", head: "", hairColor: "#49362d" },
  chef: { hair: "assets/ss14/hair-long.png", shoes: "assets/ss14/chef-shoes.png", head: "assets/ss14/chef-hat.png", hairColor: "#6b3f2c" },
  bartender: { hair: "assets/ss14/hair-business.png", shoes: "assets/ss14/color-shoes.png", head: "", hairColor: "#2b211d" },
};

const TIREDNESS_PER_SECOND = 96 / 180;
const BREAK_CHANCE_PER_SECOND = 0.03;
const BREAK_DURATION_MS = 60000;
const URGENT_DURATION_MS = 30000;
const TONE_STRESS = [0, 3, 8, 15];
const JOB_RISK_QUESTION = {
  janitor: "biohazard",
  chef: "dietary_restriction",
  bartender: "intoxication_risk",
};

let lang = "en";
let strings = {};
let jobs = [];
let lastResult = null;
let npcStates = new Map();
let simStarted = false;

/* ------------------------------------------------------------ i18n helpers */

function t(path) {
  let node = strings;
  for (const part of path.split(".")) {
    if (node == null || typeof node !== "object") return null;
    node = node[part];
  }
  return typeof node === "string" ? node : null;
}

function humanize(key) {
  return String(key).replace(/[_-]+/g, " ").trim();
}

async function setLang(next) {
  if (next !== "en" && next !== "fr") next = "en";
  const res = await fetch(`i18n/${next}.json`);
  if (!res.ok) throw new Error(`i18n ${next} failed`);
  strings = await res.json();
  lang = next;
  try {
    sessionStorage.setItem(LANG_STORAGE_KEY, lang);
  } catch (err) {
    // storage unavailable (privacy mode); language just won't persist
  }
  document.documentElement.lang = lang;
  for (const btn of els.langButtons) {
    btn.setAttribute("aria-pressed", String(btn.dataset.lang === lang));
  }
  applyStrings();
  renderJobOptions();
  renderJobDescription();
  renderCharacterProfile();
  updateCharCount();
  if (lastResult) renderResult(lastResult);
}

function applyStrings() {
  for (const el of document.querySelectorAll("[data-i18n]")) {
    const value = t(el.dataset.i18n);
    if (value != null) el.textContent = value;
  }
}

/* ----------------------------------------------------------------- loading */

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function loadJobs() {
  setStatus(t("status.loading_jobs"));
  for (let attempt = 0; attempt < JOB_LOAD_ATTEMPTS; attempt += 1) {
    try {
      const res = await fetch(`${API_BASE_URL}/jobs`);
      if (!res.ok) throw new Error(`GET /jobs -> ${res.status}`);
      jobs = await res.json();
      initNpcStates();
      renderJobOptions();
      renderJobDescription();
      renderCharacterProfile();
      els.job.disabled = jobs.length === 0;
      els.submit.disabled = jobs.length === 0;
      showError("");
      setStatus("");
      return;
    } catch (err) {
      if (attempt + 1 < JOB_LOAD_ATTEMPTS) {
        setStatus(t("status.cold_start"));
        await sleep(JOB_LOAD_RETRY_MS);
      }
    }
  }
  showError(t("errors.jobs"));
  setStatus("");
}

function renderJobOptions() {
  const current = els.job.value;
  els.job.replaceChildren();
  for (const job of jobs) {
    const opt = document.createElement("option");
    opt.value = job.key;
    const names = job.display_names || {};
    opt.textContent = names[lang] || names.en || humanize(job.key);
    els.job.appendChild(opt);
  }
  if (current && jobs.some((j) => j.key === current)) els.job.value = current;
}

function renderJobDescription() {
  const job = jobs.find((j) => j.key === els.job.value);
  const desc = job && job.description ? job.description[lang] || job.description.en : "";
  els.jobDesc.textContent = desc || "";
}

/* ------------------------------------------------------- character dossier */

function initNpcStates() {
  for (const job of jobs) {
    if (npcStates.has(job.key)) continue;
    const m = (job.profile && job.profile.mental) || {};
    npcStates.set(job.key, {
      tiredness: clamp100(m.tiredness),
      stress: clamp100(m.stress),
      onBreak: false,
      breakEndsAt: 0,
      urgentUntil: 0,
    });
  }
  if (!simStarted && jobs.length) {
    simStarted = true;
    setInterval(tickNpcStates, 1000);
  }
}

function tickNpcStates() {
  const now = Date.now();
  for (const s of npcStates.values()) {
    if (s.onBreak && now >= s.breakEndsAt) {
      s.onBreak = false;
      s.tiredness = Math.max(0, s.tiredness - 70);
      s.stress = Math.max(0, s.stress - 20);
      continue;
    }
    if (s.onBreak) continue;
    s.tiredness = Math.min(100, s.tiredness + TIREDNESS_PER_SECOND);
    if (s.tiredness > 90 && now >= s.urgentUntil
        && Math.random() < BREAK_CHANCE_PER_SECOND) {
      s.onBreak = true;
      s.breakEndsAt = now + BREAK_DURATION_MS;
    }
  }
  renderCharacterProfile();
}

function applyPredictionImpact(jobKey, answers) {
  const s = npcStates.get(jobKey);
  if (!s || !answers || typeof answers !== "object") return;
  const tone = answers.tone;
  const level = tone && Number.isInteger(tone.level) ? tone.level : 0;
  s.stress = Math.min(100, Math.max(0, s.stress + (TONE_STRESS[level] || 0)));
  const riskQid = JOB_RISK_QUESTION[jobKey];
  const risk = riskQid && answers[riskQid];
  const riskHit = risk && Number(risk.probability) >= 0.5;
  if (riskHit) s.stress = Math.min(100, s.stress + 8);
  const emergency = answers.intent && answers.intent.choice === "emergency";
  if (emergency) s.stress = Math.min(100, s.stress + 10);
  if (riskHit || emergency || level === 3) {
    s.urgentUntil = Math.max(s.urgentUntil, Date.now() + URGENT_DURATION_MS);
  }
  renderCharacterProfile();
}

function localizedField(obj) {
  return obj && typeof obj === "object" ? obj[lang] || obj.en || "" : "";
}

function clamp100(v) {
  const n = Math.round(Number(v));
  return Number.isFinite(n) ? Math.min(Math.max(n, 0), 100) : 0;
}

function renderCharacterProfile() {
  const job = jobs.find((j) => j.key === els.job.value);
  const p = job && job.profile;
  if (!job || !p) {
    els.dossier.hidden = true;
    return;
  }
  els.dossier.hidden = false;

  const role = localizedField(job.display_names) || humanize(job.key);
  els.charName.textContent = p.name || "";
  els.charRole.textContent = role;
  els.charDept.textContent = t("profile.department") || "Service";
  if (els.charEmployeeId) els.charEmployeeId.textContent = p.employee_id || "";
  if (els.charSpecies) els.charSpecies.textContent = localizedField(p.species);
  els.charAssignment.textContent = localizedField(p.assignment);
  const rt = npcStates.get(job.key);
  els.charState.textContent = rt && rt.onBreak
    ? (t("profile.on_break") || "On break")
    : localizedField(p.current_state);
  if (els.charLocation) els.charLocation.textContent = localizedField(p.location);

  const layers = SPRITE_LAYERS[job.key];
  const spriteOk =
    layers &&
    typeof p.sprite === "string" && SPRITE_RE.test(p.sprite) &&
    SPRITE_RE.test(SPRITE_BODY) &&
    SPRITE_RE.test(layers.hair) && SPRITE_RE.test(layers.shoes) &&
    (layers.head === "" || SPRITE_RE.test(layers.head));
  if (spriteOk) {
    const s = els.charSprite.style;
    s.setProperty("--body-url", `url("${SPRITE_BODY}")`);
    s.setProperty("--uniform-url", `url("${p.sprite}")`);
    s.setProperty("--shoes-url", `url("${layers.shoes}")`);
    s.setProperty("--hair-url", `url("${layers.hair}")`);
    s.setProperty("--head-url", layers.head ? `url("${layers.head}")` : "none");
    s.setProperty("--hair-color", layers.hairColor);
    els.charSprite.setAttribute("aria-label", `${p.name || job.key} — ${role}`);
    els.charSprite.hidden = false;
  } else {
    els.charSprite.hidden = true;
  }

  const tiredness = clamp100(rt ? rt.tiredness : p.mental && p.mental.tiredness);
  const stress = clamp100(rt ? rt.stress : p.mental && p.mental.stress);
  const cx = CHART.x0 + (tiredness / 100) * (CHART.x1 - CHART.x0);
  const cy = CHART.y1 - (stress / 100) * (CHART.y1 - CHART.y0);
  els.mentalPoint.setAttribute("cx", cx);
  els.mentalPoint.setAttribute("cy", cy);
  els.mentalPointGlow.setAttribute("cx", cx);
  els.mentalPointGlow.setAttribute("cy", cy);
  els.mentalGuideX.setAttribute("x1", cx);
  els.mentalGuideX.setAttribute("y1", cy);
  els.mentalGuideX.setAttribute("x2", cx);
  els.mentalGuideX.setAttribute("y2", CHART.y1);
  els.mentalGuideY.setAttribute("x1", CHART.x0);
  els.mentalGuideY.setAttribute("y1", cy);
  els.mentalGuideY.setAttribute("x2", cx);
  els.mentalGuideY.setAttribute("y2", cy);
  els.tirednessVal.textContent = String(tiredness);
  els.stressVal.textContent = String(stress);
  const tiredLabel = (t("profile.tiredness") || "Tiredness").toLowerCase();
  const stressLabel = (t("profile.stress") || "Stress").toLowerCase();
  els.mentalXLabel.textContent = t("profile.tiredness") || "Tiredness";
  els.mentalYLabel.textContent = t("profile.stress") || "Stress";
  els.mentalChart.setAttribute(
    "aria-label",
    `${t("profile.mental_health") || "Condition"}: ${tiredLabel} ${tiredness}, ${stressLabel} ${stress}`
  );
}

/* ------------------------------------------------------------ status/error */

function setStatus(text) {
  els.status.textContent = text || "";
}

function showError(text) {
  els.error.textContent = text || "";
  els.error.hidden = !text;
}

function updateCharCount() {
  const tpl = t("form.char_count");
  const n = els.message.value.length;
  els.charCount.textContent = tpl ? tpl.replace("{n}", String(n)).replace("{max}", String(MESSAGE_MAX)) : `${n}/${MESSAGE_MAX}`;
}

/* ----------------------------------------------------------------- results */

function optionLabel(key) {
  return t(`options.${key}`) || t(`levels.${key}`) || humanize(key);
}

function barRow(labelText, value, highlight) {
  const row = document.createElement("div");
  row.className = "bar-row" + (highlight ? " is-chosen" : "");

  const label = document.createElement("span");
  label.className = "bar-label";
  label.textContent = labelText;

  const track = document.createElement("div");
  track.className = "bar-track";
  track.setAttribute("role", "img");
  track.setAttribute("aria-label", `${labelText}: ${(value * 100).toFixed(1)}%`);
  const fill = document.createElement("div");
  fill.className = "bar-fill";
  fill.style.width = `${Math.min(Math.max(value, 0), 1) * 100}%`;
  track.appendChild(fill);

  const pct = document.createElement("span");
  pct.className = "bar-value";
  pct.textContent = `${(value * 100).toFixed(1)}%`;

  row.append(label, track, pct);
  return row;
}

function questionTitle(qid) {
  return t(`questions.${qid}`) || humanize(qid);
}

function renderAnswerCard(qid, answer) {
  const card = document.createElement("article");
  card.className = "answer-card";

  const title = document.createElement("h3");
  title.textContent = questionTitle(qid);
  card.appendChild(title);

  if (answer.source === "phrase_rule") {
    const badge = document.createElement("span");
    badge.className = "phrase-badge";
    badge.textContent = t("results.phrase_rule") || "Exact phrase rule";
    card.appendChild(badge);
  }

  if (answer.type === "choice" || answer.type === "score") {
    const chosenText =
      answer.type === "choice"
        ? optionLabel(answer.choice)
        : t(`levels.${answer.level}`) || answer.label || `${t("results.level_prefix") || "Level"} ${answer.level}`;

    const chosen = document.createElement("p");
    chosen.className = "chosen";
    const chosenLabel = document.createElement("span");
    chosenLabel.className = "chosen-key";
    chosenLabel.textContent = `${t("results.chosen") || "Answer"}: `;
    const chosenValue = document.createElement("strong");
    chosenValue.textContent = chosenText;
    const conf = document.createElement("span");
    conf.className = "confidence";
    conf.textContent = ` (${t("results.confidence") || "confidence"} ${(answer.confidence * 100).toFixed(1)}%)`;
    chosen.append(chosenLabel, chosenValue, conf);
    card.appendChild(chosen);

    const bars = document.createElement("div");
    bars.className = "bars";
    // Natural order: the order the backend serialized probabilities in.
    for (const [key, value] of Object.entries(answer.probabilities || {})) {
      const labelText = answer.type === "score"
        ? (t(`levels.${key}`) || `${t("results.level_prefix") || "Level"} ${key}`)
        : optionLabel(key);
      const isChosen = answer.type === "choice" ? key === answer.choice : key === String(answer.level);
      bars.appendChild(barRow(labelText, Number(value) || 0, isChosen));
    }
    card.appendChild(bars);
  } else if (answer.type === "noul") {
    const p = Number(answer.probability) || 0;
    const bars = document.createElement("div");
    bars.className = "bars";
    bars.appendChild(barRow(t("results.yes") || "Yes", p, p >= 0.5));
    card.appendChild(bars);
  }
  return card;
}

function renderResult(result) {
  els.answers.replaceChildren();
  if (result.reply) {
    const block = document.createElement("div");
    block.className = "npc-reply";
    const label = document.createElement("p");
    label.className = "npc-reply-label";
    label.textContent = t("results.reply") || "NPC reply";
    const text = document.createElement("p");
    text.className = "npc-reply-text";
    text.textContent = result.reply;
    block.append(label, text);
    els.answers.appendChild(block);
  }
  for (const [qid, answer] of Object.entries(result.answers || {})) {
    els.answers.appendChild(renderAnswerCard(qid, answer));
  }
  els.requestId.textContent = `${t("results.request_id") || "Request"}: ${result.request_id || ""}`;
  els.results.hidden = false;
}

/* ------------------------------------------------------------------ submit */

els.job.addEventListener("change", () => {
  renderJobDescription();
  renderCharacterProfile();
});
els.message.addEventListener("input", updateCharCount);

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");

  const message = els.message.value.trim();
  if (!els.job.value || message.length < 1 || message.length > MESSAGE_MAX) {
    showError(t("errors.validation"));
    return;
  }
  const jobKey = els.job.value;

  els.submit.disabled = true;
  els.submit.textContent = t("form.submitting") || "…";
  setStatus(t("status.sending"));

  let settled = false;
  const coldTimer = setTimeout(() => {
    if (!settled) setStatus(t("status.cold_start"));
  }, COLD_START_DELAY_MS);

  const started = performance.now();
  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job: jobKey, message, lang }),
    });
    if (res.status === 429) {
      setStatus("");
      showError(t("errors.rate_limited"));
      return;
    }
    if (!res.ok) {
      setStatus("");
      showError(t("errors.server"));
      return;
    }
    lastResult = await res.json();
    const totalMs = performance.now() - started;
    applyPredictionImpact(jobKey, lastResult.answers);
    renderResult(lastResult);
    let statusText = `${t("status.done")} ${(totalMs / 1000).toFixed(1)} s`;
    if (lastResult.latency_ms != null) {
      statusText += ` (${t("status.server")} ${(lastResult.latency_ms / 1000).toFixed(1)} s)`;
    }
    if (totalMs - (lastResult.latency_ms ?? 0) > COLD_START_GAP_MS) {
      statusText += ` · ${t("status.cold_start_tag")}`;
    }
    setStatus(statusText);
  } catch (err) {
    setStatus("");
    showError(t("errors.network"));
    // message textarea is never cleared, so nothing the user typed is lost
  } finally {
    settled = true;
    clearTimeout(coldTimer);
    els.submit.disabled = false;
    els.submit.textContent = t("form.submit") || "Ask";
  }
});

for (const btn of els.langButtons) {
  btn.addEventListener("click", () => {
    setLang(btn.dataset.lang).catch(() => {});
  });
}

/* --------------------------------------------------------------------- init */

(async function init() {
  let stored = null;
  try {
    stored = sessionStorage.getItem(LANG_STORAGE_KEY);
  } catch (err) {
    // storage unavailable; fall back to the browser language
  }
  const initial = stored === "en" || stored === "fr"
    ? stored
    : (navigator.language || "en").toLowerCase().startsWith("fr") ? "fr" : "en";
  try {
    await setLang(initial);
  } catch (err) {
    // fall back to whatever is already in the DOM
  }
  updateCharCount();
  loadJobs();
})();
