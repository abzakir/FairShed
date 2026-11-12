"""Interactive "city at night" view of a load-shedding plan.

Each feeder is drawn as a small district. Its windows glow while it has power
and go dark in the hours it is switched off. The viewer can step through the
hours, press play, and switch between plans (for example FairShed against the
priority list) to see who sits in the dark and for how long.

The view is one self-contained HTML document (no external scripts or fonts),
shown in the Streamlit app through an iframe.
"""

from __future__ import annotations

import json

from .instance import Instance
from .schedule import schedule_matrix, slot_range

_KINDS = (
    ("hospital", ("hospital", "clinic", "medical", "water", "health")),
    ("industrial", ("industr", "factory", "estate", "mill", "plant")),
    ("commercial", ("commerc", "market", "shop", "bazaar", "mall", "plaza")),
    ("residential", ("resid", "block", "colony", "town", "housing", "home", "society")),
)


def feeder_kind(name: str) -> str:
    """Pick the district drawing from words in the feeder's name."""
    lowered = name.lower()
    for kind, words in _KINDS:
        if any(word in lowered for word in words):
            return kind
    return "tower"


def city_view_html(
    inst: Instance,
    plans: list[tuple[str, int, str]],
    protected: tuple[str, ...] = (),
    title: str = "Tonight on the grid",
) -> str:
    """plans: list of (label, schedule index, one-line note). The first is shown first."""
    feeders = [{"name": fd.name, "kind": feeder_kind(fd.name), "protected": False}
               for fd in inst.feeders]
    feeders += [{"name": name, "kind": feeder_kind(name), "protected": True} for name in protected]
    plan_data = []
    for label, index, note in plans:
        matrix = schedule_matrix(inst, index).tolist()
        matrix += [[0] * inst.n_slots for _ in protected]
        plan_data.append({"label": label, "note": note, "matrix": matrix})
    data = {
        "title": title,
        "slots": [slot_range(label) for label in inst.slot_labels],
        "need": [d * inst.unit_mw for d in inst.deficits],
        "unit": inst.unit_mw,
        "feeders": feeders,
        "plans": plan_data,
    }
    # "</" must not appear inside a <script> block; feeder names are user input.
    payload = json.dumps(data).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", payload)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<style>
  :root { --ink:#eef2fb; --muted:#9aa8c4; --amber:#ffc857; --line:#ffffff1f; }
  * { box-sizing:border-box; margin:0; }
  html, body { background:transparent; }
  body { font-family:"Inter","Segoe UI",system-ui,-apple-system,sans-serif; color:var(--ink); }
  .scene { position:relative; overflow:hidden; border-radius:18px; padding:20px 22px 18px;
           background:radial-gradient(900px 320px at 88% -12%, #2f55a855, transparent 62%),
                      linear-gradient(180deg,#0a1226 0%,#12203f 70%,#182b55 100%); }
  .star { position:absolute; width:2px; height:2px; border-radius:50%; background:#fff; opacity:.5;
          animation:twinkle 3.2s ease-in-out infinite; }
  @keyframes twinkle { 0%,100% { opacity:.15 } 50% { opacity:.85 } }
  .moon { position:absolute; right:34px; top:22px; width:34px; height:34px; border-radius:50%;
          background:radial-gradient(circle at 35% 35%, #fff8e1, #f3dfa2 70%);
          box-shadow:0 0 38px 8px #f3dfa244; opacity:.9; }
  .head { position:relative; display:flex; justify-content:space-between; align-items:flex-start;
          gap:14px; flex-wrap:wrap; padding-right:64px; }
  h2 { font-size:18px; font-weight:700; letter-spacing:.01em; }
  .sub { font-size:13.5px; color:var(--ink); margin-top:4px; font-variant-numeric:tabular-nums; }
  .note { font-size:12.5px; color:var(--muted); margin-top:3px; min-height:16px; }
  .toggle { display:inline-flex; background:#ffffff14; border-radius:999px; padding:3px; }
  .toggle button { border:0; background:transparent; color:var(--muted); cursor:pointer;
                   font:600 12.5px/1 inherit; font-family:inherit; padding:8px 14px; border-radius:999px;
                   transition:background .25s,color .25s; }
  .toggle button[aria-pressed="true"] { background:#fff; color:#0f1b33; }
  .districts { position:relative; display:grid; grid-auto-flow:column;
               grid-auto-columns:minmax(150px,1fr); gap:14px; margin:16px 0 14px;
               overflow-x:auto; padding-bottom:2px; }
  .d { background:#ffffff0d; border:1px solid var(--line); border-radius:14px; padding:12px 12px 11px;
       transition:background .6s,border-color .6s,box-shadow .6s; }
  .d.on { border-color:#ffc85752; box-shadow:inset 0 -30px 40px -34px #ffc857aa; }
  .d.off { background:#00000026; }
  .d svg { width:100%; height:86px; display:block; }
  .bld { fill:#3a4f80; transition:fill .7s; }     .d.off .bld { fill:#1c2947; }
  .roof { fill:#4a63a0; transition:fill .7s; }    .d.off .roof { fill:#222f52; }
  .win { fill:var(--amber); transition:fill .7s; } .d.off .win { fill:#141d38; }
  .door { fill:#22325a; }
  .awn { fill:#5aa2ff; transition:fill .7s; }     .d.off .awn { fill:#2a3a63; }
  .sign { fill:#e8f1ff; transition:opacity .7s; } .d.off .sign, .d.off .signtxt { opacity:.3; }
  .signtxt { fill:#1c5cab; font:700 13px sans-serif; transition:opacity .7s; }
  .halo { fill:var(--amber); opacity:.22; transition:opacity .7s; } .d.off .halo { opacity:0; }
  .smoke { fill:#c9d4ec; opacity:.45; transition:opacity .7s; }     .d.off .smoke { opacity:0; }
  .ground { fill:#ffffff24; }
  .name { font-weight:650; font-size:13.5px; margin-top:8px; white-space:nowrap; overflow:hidden;
          text-overflow:ellipsis; }
  .state { display:inline-block; font-size:12px; font-weight:600; margin-top:6px; padding:4px 10px;
           border-radius:999px; transition:background .5s,color .5s; }
  .d.on .state { background:#ffc85726; color:var(--amber); }
  .d.off .state { background:#ffffff14; color:#b9c4dc; }
  .dots { display:flex; gap:5px; margin-top:10px; }
  .dot { flex:1; height:6px; border-radius:3px; background:var(--amber); }
  .dot.dark { background:#3a4a70; }
  .dot.now { box-shadow:0 0 0 2px #0f1b33, 0 0 0 3.5px #fff; }
  .tally { font-size:11.5px; color:var(--muted); margin-top:7px; }
  .timeline { position:relative; display:flex; align-items:stretch; gap:10px; }
  .play { flex:0 0 44px; border:0; border-radius:12px; background:var(--amber); color:#1a1300;
          font-size:15px; cursor:pointer; transition:transform .15s,filter .15s; }
  .play:hover { filter:brightness(1.08); transform:translateY(-1px); }
  .hours { display:flex; gap:8px; flex:1; }
  .hour { flex:1; border:1px solid var(--line); background:#ffffff0d; color:var(--ink);
          border-radius:12px; padding:8px 6px; cursor:pointer; text-align:center;
          font:600 12.5px/1.2 inherit; font-family:inherit; transition:background .25s,color .25s; }
  .hour small { display:block; font-weight:500; color:var(--muted); margin-top:3px; font-size:11px; }
  .hour:hover { background:#ffffff1a; }
  .hour[aria-pressed="true"] { background:#fff; color:#0f1b33; border-color:#fff; }
  .hour[aria-pressed="true"] small { color:#42506d; }
  button:focus-visible { outline:2px solid #fff; outline-offset:2px; }
  @media (prefers-reduced-motion: reduce) { .star { animation:none; } * { transition:none !important; } }
</style></head>
<body>
<div class="scene">
  <div class="moon" aria-hidden="true"></div>
  <div class="head">
    <div>
      <h2 id="title"></h2>
      <div class="sub" id="sub" aria-live="polite"></div>
      <div class="note" id="note"></div>
    </div>
    <div class="toggle" id="toggle" role="group" aria-label="Plan to show"></div>
  </div>
  <div class="districts" id="districts"></div>
  <div class="timeline">
    <button class="play" id="play" aria-label="Play through the hours">&#9654;</button>
    <div class="hours" id="hours" role="group" aria-label="Hour to show"></div>
  </div>
</div>
<script>
const D = __DATA__;
const $ = (s) => document.querySelector(s);
const GROUND = '<ellipse class="halo" cx="80" cy="86" rx="72" ry="9"/><rect class="ground" x="0" y="86" width="160" height="2.5" rx="1"/>';
const wins = (xs, ys, w, h) => ys.map(y => xs.map(x => `<rect class="win" x="${x}" y="${y}" width="${w}" height="${h}" rx="1"/>`).join("")).join("");
const ART = {
  hospital: `<rect class="bld" x="14" y="54" width="26" height="32" rx="2"/><rect class="bld" x="120" y="54" width="26" height="32" rx="2"/>
    <rect class="bld" x="36" y="30" width="88" height="56" rx="3"/>
    <rect class="sign" x="68" y="9" width="24" height="20" rx="4"/><text class="signtxt" x="80" y="24" text-anchor="middle">H</text>
    ${wins([45, 61, 77, 93, 109], [39, 54], 8, 9)}${wins([21, 127], [62], 10, 8)}
    <rect class="door" x="73" y="70" width="14" height="16" rx="1"/>`,
  industrial: `<circle class="smoke" cx="119" cy="9" r="5"/><circle class="smoke" cx="127" cy="4" r="4"/><circle class="smoke" cx="136" cy="16" r="4"/>
    <rect class="bld" x="112" y="16" width="10" height="44"/><rect class="bld" x="129" y="26" width="10" height="34"/>
    <polygon class="roof" points="14,58 14,44 38,58 38,44 62,58 62,44 86,58 86,44 106,58"/>
    <rect class="bld" x="14" y="58" width="132" height="28" rx="2"/>
    ${wins([22, 42, 62, 82, 102, 124], [65], 13, 10)}`,
  commercial: `<rect class="bld" x="12" y="48" width="42" height="38" rx="2"/><rect class="bld" x="59" y="36" width="42" height="50" rx="2"/><rect class="bld" x="106" y="48" width="42" height="38" rx="2"/>
    <rect class="awn" x="10" y="46" width="46" height="7" rx="2"/><rect class="awn" x="57" y="34" width="46" height="7" rx="2"/><rect class="awn" x="104" y="46" width="46" height="7" rx="2"/>
    ${wins([19], [60], 28, 15)}${wins([66], [48], 28, 12)}${wins([66], [65], 28, 14)}${wins([113], [60], 28, 15)}`,
  residential: `<polygon class="roof" points="8,56 31,38 54,56"/><polygon class="roof" points="57,50 80,29 103,50"/><polygon class="roof" points="106,56 129,38 152,56"/>
    <rect class="bld" x="12" y="56" width="38" height="30"/><rect class="bld" x="61" y="50" width="38" height="36"/><rect class="bld" x="110" y="56" width="38" height="30"/>
    ${wins([17, 36], [62], 9, 9)}${wins([66, 85], [57], 9, 9)}${wins([115, 134], [62], 9, 9)}
    <rect class="door" x="27" y="74" width="8" height="12"/><rect class="door" x="76" y="72" width="8" height="14"/><rect class="door" x="125" y="74" width="8" height="12"/>`,
  tower: `<rect class="bld" x="28" y="20" width="44" height="66" rx="2"/><rect class="bld" x="84" y="38" width="48" height="48" rx="2"/>
    ${wins([35, 46, 57], [28, 42, 56, 70], 8, 8)}${wins([91, 104, 117], [46, 60, 72], 9, 8)}`,
};
let plan = 0, hour = 0, timer = null;

function build() {
  $("#title").textContent = D.title;
  const scene = $(".scene");
  for (let i = 0; i < 46; i++) {
    const s = document.createElement("span");
    s.className = "star";
    s.style.left = (Math.random() * 100) + "%";
    s.style.top = (Math.random() * 46) + "%";
    s.style.animationDelay = (Math.random() * 3) + "s";
    scene.prepend(s);
  }
  const toggle = $("#toggle");
  if (D.plans.length > 1) {
    D.plans.forEach((p, i) => {
      const b = document.createElement("button");
      b.textContent = p.label;
      b.onclick = () => { plan = i; render(); };
      toggle.appendChild(b);
    });
  } else { toggle.style.display = "none"; }
  D.feeders.forEach((f) => {
    const d = document.createElement("div");
    d.className = "d on";
    d.innerHTML = `<svg viewBox="0 0 160 90" preserveAspectRatio="xMidYMax meet" aria-hidden="true">${GROUND}${ART[f.kind] || ART.tower}</svg>
      <div class="name"></div><div class="state"></div><div class="dots"></div><div class="tally"></div>`;
    d.querySelector(".name").textContent = f.name;
    d.querySelector(".name").title = f.name;
    $("#districts").appendChild(d);
  });
  D.slots.forEach((label, t) => {
    const b = document.createElement("button");
    b.className = "hour";
    const top = document.createElement("span"); top.textContent = label;
    const low = document.createElement("small"); low.textContent = D.need[t] + " MW short";
    b.append(top, low);
    b.onclick = () => { stop(); hour = t; render(); };
    $("#hours").appendChild(b);
  });
  $("#play").onclick = () => (timer ? stop() : play(true));
  render();
  if (!matchMedia("(prefers-reduced-motion: reduce)").matches) play(false);
}

function render() {
  const P = D.plans[plan], M = P.matrix;
  [...$("#toggle").children].forEach((b, i) => b.setAttribute("aria-pressed", i === plan));
  let dark = 0;
  [...$("#districts").children].forEach((d, i) => {
    const off = !!M[i][hour];
    if (off) dark++;
    d.classList.toggle("off", off);
    d.classList.toggle("on", !off);
    d.querySelector(".state").textContent = D.feeders[i].protected ? "🛡 Protected · on"
      : (off ? "🌙 Off this hour" : "⚡ Power on");
    const dots = d.querySelector(".dots");
    dots.replaceChildren(...M[i].map((v, t) => {
      const s = document.createElement("span");
      s.className = "dot" + (v ? " dark" : "") + (t === hour ? " now" : "");
      return s;
    }));
    const total = M[i].reduce((a, b) => a + b, 0);
    d.querySelector(".tally").textContent = total ? `off ${total} of ${M[i].length} h tonight` : "on all evening";
  });
  [...$("#hours").children].forEach((b, t) => b.setAttribute("aria-pressed", t === hour));
  $("#sub").textContent = `${D.slots[hour]} · ${dark * D.unit} of ${D.need[hour]} MW shed · ${dark} feeder${dark === 1 ? "" : "s"} dark`;
  $("#note").textContent = P.note || "";
}

// loop = true keeps cycling; loop = false plays the evening once, then rests on the first hour.
function play(loop) {
  $("#play").innerHTML = "&#10074;&#10074;";
  $("#play").setAttribute("aria-label", "Pause");
  timer = setInterval(() => {
    const next = hour + 1;
    if (next >= D.slots.length && !loop) { stop(); hour = 0; render(); return; }
    hour = next % D.slots.length;
    render();
  }, 1600);
}
function stop() {
  clearInterval(timer); timer = null;
  $("#play").innerHTML = "&#9654;";
  $("#play").setAttribute("aria-label", "Play through the hours");
}
build();
</script>
</body></html>
"""
