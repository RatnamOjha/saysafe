// earshot demo page: renders trace events from /ws. No framework.
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  thresholds: { t_accept: 0.45, t_reject: 0.25 },
  scenes: [],
  scene: 0,
  mic: false,
  flags: { headphones: false, discreet_mode: false },
  led: "idle",
  ledSince: performance.now(),
  lastVoice: null, // {label, score, at}
  approvalCard: null,
  replyCard: null,
  replyLine: null,
};

// ---------- helpers ----------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmt = (x, d = 2) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
const post = (url, body) =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function clearEmpty(container) {
  const empty = container.querySelector(".empty");
  if (empty) empty.remove();
}

// ---------- ring ----------
const LED_COLORS = {
  idle: [138, 150, 166, 0.25], off: [138, 150, 166, 0.12], listening: [240, 244, 248, 1],
  working: [143, 180, 255, 1], done: [94, 224, 160, 1], amber: [255, 181, 71, 1], refused: [255, 107, 107, 1],
};
const LED_LABELS = {
  idle: "idle", off: "off", listening: "listening", working: "working",
  done: "done", amber: "needs your phone", refused: "refused",
};

function setLed(s) {
  state.led = LED_COLORS[s] ? s : "idle";
  state.ledSince = performance.now();
  $("led-label").textContent = LED_LABELS[state.led];
}

function drawRing(now) {
  const c = $("ring");
  const g = c.getContext("2d");
  const w = c.width, r = w * 0.38, cx = w / 2, cy = w / 2;
  g.clearRect(0, 0, w, w);
  const [R, G, B, A] = LED_COLORS[state.led];
  const t = (now - state.ledSince) / 1000;

  g.lineWidth = 34;
  g.strokeStyle = "rgba(255,255,255,0.05)";
  g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.stroke();

  let alpha = A;
  if (state.led === "listening") alpha = 0.45 + 0.55 * (0.5 + 0.5 * Math.sin(t * 3.2));
  if (state.led === "done" || state.led === "refused") alpha = Math.max(0.35, 1 - Math.max(0, t - 2) * 0.3);
  if (state.led === "amber") alpha = 0.6 + 0.4 * (0.5 + 0.5 * Math.sin(t * 2));

  g.shadowColor = `rgba(${R},${G},${B},${alpha})`;
  g.shadowBlur = state.led === "idle" || state.led === "off" ? 0 : 40;
  g.strokeStyle = `rgba(${R},${G},${B},${alpha})`;
  g.lineCap = "round";
  g.beginPath();
  if (state.led === "working") {
    const start = t * 4;
    g.arc(cx, cy, r, start, start + Math.PI * 0.6);
  } else {
    g.arc(cx, cy, r, 0, Math.PI * 2);
  }
  g.stroke();
  g.shadowBlur = 0;
  requestAnimationFrame(drawRing);
}
requestAnimationFrame(drawRing);

// ---------- transcript ----------
function voiceTag(label, score) {
  if (!label) return ["sys", "Voice"];
  if (label === "owner") return ["you", `You ${fmt(score)}`];
  if (label === "other") return ["other", `Not you ${fmt(score)}`];
  return ["unclear", score === null || score === undefined ? "Unclear" : `Unclear ${fmt(score)}`];
}

function addLine(cls, tag, text, extra = "") {
  const ol = $("lines");
  const li = el("li", extra, `<span class="tag ${cls}">${esc(tag)}</span><span>${esc(text)}</span>`);
  ol.appendChild(li);
  ol.scrollTop = ol.scrollHeight;
  return li;
}

function setLineTag(li, cls, tag) {
  if (!li) return;
  const span = li.querySelector(".tag");
  span.className = `tag ${cls}`;
  span.textContent = tag;
}

// ---------- why cards ----------
function newCard(title) {
  const cards = $("cards");
  clearEmpty(cards);
  const card = el("div", "card", `<h3>${esc(title)}</h3>`);
  cards.prepend(card);
  while (cards.children.length > 3) cards.lastChild.remove();
  return card;
}

function addRow(card, key, html) {
  if (!card) return;
  card.appendChild(el("div", "row", `<span class="k">${esc(key)}</span><span>${html}</span>`));
}

function scoreBar(card, s) {
  const t = state.thresholds;
  const pct = (x) => `${Math.max(0, Math.min(1, x)) * 100}%`;
  const bar = el("div", "bar");
  bar.style.setProperty("--rej", pct(t.t_reject));
  bar.style.setProperty("--acc", pct(t.t_accept));
  for (const x of [t.t_reject, t.t_accept]) {
    const tick = el("div", "tick"); tick.style.left = pct(x); bar.appendChild(tick);
  }
  for (const kind of ["command", "reply", "fused"]) {
    if (s[kind] === null || s[kind] === undefined) continue;
    const dot = el("div", `dot ${kind}`); dot.style.left = pct(s[kind]); dot.title = `${kind} ${fmt(s[kind])}`;
    bar.appendChild(dot);
  }
  card.appendChild(bar);
  card.appendChild(el("div", "legend",
    `<span>reject &lt; ${fmt(t.t_reject)}</span><span>accept ≥ ${fmt(t.t_accept)}</span>` +
    `<span>● fused ${fmt(s.fused)}</span><span style="color:var(--accent)">● reply ${fmt(s.reply)}</span>` +
    `<span>● command ${fmt(s.command)}</span>`));
}

function actionSummary(a) {
  const amount = a.amount ? ` $${a.amount}` : "";
  const who = a.counterparty ? ` ${a.counterparty}` : "";
  return `${a.type.replace(/_/g, " ")}${who}${amount}`;
}

function highlight(text, spans) {
  let out = "", i = 0;
  for (const s of [...spans].sort((a, b) => a.start - b.start)) {
    if (s.start < i) continue;
    out += esc(text.slice(i, s.start)) + `<mark title="${esc(s.category)}">${esc(text.slice(s.start, s.end))}</mark>`;
    i = s.end;
  }
  return out + esc(text.slice(i));
}

// ---------- phone ----------
function phoneMessage(e) {
  const feed = $("phone-feed");
  clearEmpty(feed);
  const n = el("div", "notif", `<div class="title">${esc(e.title || "earshot")}</div><div>${esc(e.text)}</div>`);
  if (e.action_id) {
    const buttons = el("div", "buttons");
    const ok = el("button", "", "Approve"), no = el("button", "deny", "Deny");
    const settle = async (verb) => {
      ok.disabled = no.disabled = true;
      const r = await post(`/approvals/${e.action_id}/${verb}`);
      const body = await r.json().catch(() => ({}));
      buttons.replaceWith(el("div", "done", r.ok ? (verb === "approve" ? "Approved" : "Denied") : esc(body.detail || "Expired")));
    };
    ok.onclick = () => settle("approve");
    no.onclick = () => settle("deny");
    buttons.append(ok, no);
    n.appendChild(buttons);
  }
  feed.prepend(n);
}

// ---------- status chips ----------
function renderChips(aud) {
  const mic = $("chip-mic");
  mic.textContent = state.mic ? "Mic on" : "Mic off";
  mic.className = `chip ${state.mic ? "good" : "off"}`;
  $("chip-headphones").className = `chip ${state.flags.headphones ? "on" : "off"}`;
  $("chip-discreet").className = `chip ${state.flags.discreet_mode ? "on" : "off"}`;
  if (aud) {
    const a = $("chip-audience");
    if (aud.level === "others_present") {
      a.textContent = `Another voice ${Math.round(aud.seconds_since_other ?? 0)} s ago`;
      a.className = "chip bad";
    } else if (aud.level === "alone_likely") {
      a.textContent = "Alone";
      a.className = "chip good";
    } else {
      a.textContent = aud.listening_seconds > 0 ? `Listening for ${Math.round(aud.listening_seconds)} s` : "Room unknown";
      a.className = "chip warn";
    }
  }
}

function renderScene() {
  const s = state.scenes.find((x) => x.key === state.scene);
  $("scene-title").textContent = s ? `Scene ${s.key}: ${s.title}` : "";
  $("scene-lines").textContent = s ? s.lines : "Press 1–7 to pick a scene.";
}

// ---------- events ----------
const handlers = {
  hello(e) {
    state.thresholds = e.thresholds; state.scenes = e.scenes; state.scene = e.scene;
    state.mic = e.mic; state.flags = e.flags;
    renderScene(); renderChips(e.audience);
  },
  audience_tick(e) { renderChips(e); },
  mic(e) { state.mic = e.on; renderChips(); },
  flags(e) { state.flags = { headphones: e.headphones, discreet_mode: e.discreet_mode }; renderChips(); },
  led(e) { setLed(e.state); },
  reset(e) {
    state.scene = e.scene; renderScene();
    $("lines").innerHTML = ""; $("cards").innerHTML = '<p class="empty">Decisions show up here.</p>';
    $("phone-feed").innerHTML = '<p class="empty">Nothing yet.</p>'; $("caption").textContent = "…";
    state.approvalCard = state.replyCard = null;
  },
  voice(e) { state.lastVoice = { label: e.label, score: e.score, at: performance.now() }; },
  transcript(e) {
    const v = state.lastVoice && performance.now() - state.lastVoice.at < 4000 ? state.lastVoice : null;
    const [cls, tag] = v ? voiceTag(v.label, v.score) : ["sys", "Typed"];
    addLine(cls, tag, e.text);
    state.lastVoice = null;
  },
  ignored(e) { addLine("sys", "Ignored", "Not a request, so the band stays quiet.", "dim"); },
  reply_captured(e) {
    state.replyLine = addLine("sys", e.timed_out ? "No reply" : "Reply", e.timed_out ? "(silence)" : e.text);
  },
  spoken(e) {
    $("caption").textContent = e.text;
    addLine("band", e.channel === "headphones" ? "Band (headphones)" : "Band", e.text);
  },
  phone(e) { phoneMessage(e); },

  risk_assessed(e) {
    const c = newCard("Approval");
    state.approvalCard = c;
    addRow(c, "Action", esc(actionSummary(e.action)));
    addRow(c, "Tier", `<span class="pill">${esc(e.tier)}</span>${e.bumped ? " (raised one tier)" : ""}`);
    addRow(c, "Rule", esc(e.rule_id));
    addRow(c, "Why", esc(e.reasons.join("; ")));
  },
  readback_spoken(e) {
    let html = esc(e.text);
    if (e.challenge_word) html = html.replace(esc(e.challenge_word), `<span class="word">${esc(e.challenge_word)}</span>`);
    addRow(state.approvalCard, e.private ? "Kept private" : "Read-back", html);
  },
  speaker_scored(e) {
    const c = state.approvalCard;
    if (!c) return;
    if (!e.owner_enrolled) addRow(c, "Voice", "No owner enrolled");
    else if (e.fused === null) addRow(c, "Voice", "Too little speech to score");
    else { addRow(c, "Voice", `fused ${fmt(e.fused)} → <b>${esc(e.band)}</b>`); scoreBar(c, e); }
    if (e.reply !== null && e.reply !== undefined) {
      const label = e.reply >= state.thresholds.t_accept ? "owner" : e.reply < state.thresholds.t_reject ? "other" : "unclear";
      const [cls, tag] = voiceTag(label, e.reply);
      setLineTag(state.replyLine, cls, tag);
    }
  },
  reply_matched(e) { addRow(state.approvalCard, "Reply", `${esc(e.kind.replace(/_/g, " "))} <span class="lat">${esc(e.detail)}</span>`); },
  decision(e) {
    const c = state.approvalCard;
    if (!c) return;
    addRow(c, "Decision", `<b class="out-${e.outcome}">${esc(e.outcome.replace("_", " "))}</b> via ${esc(e.method)}`);
    const steps = e.latency_ms_by_step || {};
    const parts = Object.entries(steps).map(([k, v]) => `${k} ${Math.round(v)} ms`);
    if (parts.length) addRow(c, "Latency", `<span class="lat">${esc(parts.join(" · "))}</span>`);
  },
  executed() { if (state.approvalCard) addRow(state.approvalCard, "Result", '<b class="out-approve">executed</b>'); },
  refused(e) { if (state.approvalCard) addRow(state.approvalCard, "Result", `<b class="out-reject">refused: ${esc(e.reason)}</b>`); },

  detection(e) {
    const c = newCard("Reply");
    state.replyCard = c;
    c.dataset.text = "";
    addRow(c, "Level", `<span class="pill lvl-${e.level}">${esc(e.level)}</span> ${esc((e.categories || []).join(", "))}`);
    c._spans = e.spans || [];
  },
  audience_state(e) {
    const c = state.replyCard;
    renderChips(e);
    if (!c) return;
    addRow(c, "Room", esc(e.level.replace(/_/g, " ")) + ` <span class="lat">${esc((e.evidence || []).join("; "))}</span>`);
  },
  route_decision(e) {
    const c = state.replyCard;
    if (!c) return;
    addRow(c, "Rule", esc(e.rule_cell));
    addRow(c, "Channel", `<b>${esc(e.channel.replace(/_/g, " "))}</b>`);
    if (e.reasons && e.reasons.length) addRow(c, "Why", esc(e.reasons.join("; ")));
  },
  route(e) {
    const c = state.replyCard;
    if (!c) return;
    const original = e.phone || e.spoken || "";
    if (e.phone) {
      c.appendChild(el("div", "said", `<span class="k">Said</span> ${esc(e.spoken || "(nothing)")}`));
      c.appendChild(el("div", "sent", `<span class="k">Phone</span> ${highlight(original, c._spans || [])}`));
    } else if (e.spoken) {
      c.appendChild(el("div", "said", `<span class="k">Said</span> ${highlight(e.spoken, c._spans || [])}`));
    }
  },
  rewrite_step(e) { addRow(state.replyCard, "Rewrite", esc(e.step)); },
  error(e) { addLine("other", "Error", e.message); },
};

// ---------- socket ----------
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { $("conn").textContent = "Connected"; $("conn").className = "chip good"; };
  ws.onclose = () => { $("conn").textContent = "Disconnected"; $("conn").className = "chip bad"; setTimeout(connect, 1000); };
  ws.onmessage = (m) => {
    const e = JSON.parse(m.data);
    const h = handlers[e.type];
    if (h) {
      try { h(e); } catch (err) { console.error(e.type, err); }
    }
  };
}
connect();

// ---------- keys and text ----------
document.addEventListener("keydown", (ev) => {
  if (ev.target === $("text-input") || ev.metaKey || ev.ctrlKey) return;
  const k = ev.key.toLowerCase();
  if (k === "m") post("/mic", { on: !state.mic });
  else if (k === "h") post("/flags", { headphones: !state.flags.headphones });
  else if (k === "d") post("/flags", { discreet_mode: !state.flags.discreet_mode });
  else if (k === "r") post("/reset", {});
  else if (/^[1-7]$/.test(k)) post("/reset", { scene: Number(k) });
  else if (k === "/" || k === "t") { ev.preventDefault(); $("text-input").focus(); }
});

$("text-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const text = $("text-input").value.trim();
  if (text) post("/text", { text });
  $("text-input").value = "";
  $("text-input").blur();
});
