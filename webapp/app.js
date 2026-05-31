/* ============================================================================
   Toronto Ambulance Relocation — Gotham-style command frontend
   Stage 1: spinning globe (globe.gl) -> flyTo Toronto
   Stage 2: deck.gl dark map driven by either
            - LIVE backend (/api/step: sim + cuOpt + Nemotron), or
            - a pre-recorded timeline.json  (fallback when no backend)
============================================================================ */
const TICK_MS = 2000;          // playback / live cadence (ms per decision tick)
const TORONTO = { lat: 43.70, lng: -79.39 };
const $ = (id) => document.getElementById(id);

let WORLD, GEO, MAXDEMAND = 1;
let MODE = "static";           // "live" | "static"
let TIMELINE = null, idx = 0;  // static mode
let deckgl, CURRENT = null, RSEQ = 0;
let playing = false, timer = null, busy = false, pendingCommand = null;
let FEED = [];                 // accumulated incident feed (killfeed)

// ---- colors ---------------------------------------------------------------
function coverageColor(fsa) {
  // structurally uncoverable (no station can ever reach it) -> distinct slate/violet,
  // NOT the red used for a transient gap. Should be empty after the data fix.
  if (WORLD && WORLD.fsa_uncoverable && WORLD.fsa_uncoverable.includes(fsa))
    return [120, 95, 165, 150];
  const c = CURRENT && CURRENT.coverage[fsa];
  if (!c) return [90, 110, 120, 18];
  const a = 45 + 165 * Math.min(1, c.demand / MAXDEMAND);
  return c.covered ? [37, 224, 160, a] : [255, 60, 60, a];
}
const STATUS_COLOR = (s) => (s === "available" ? [37, 224, 160] : [245, 165, 36]);
// 911 priority -> blip color (Delta/Echo = life-threatening = red)
const PRIORITY_COLOR = (p) =>
  ({ DELTA: [255, 60, 60], ECHO: [255, 60, 60], CHARLIE: [245, 165, 36],
     BRAVO: [245, 200, 60], ALPHA: [37, 224, 160] }[p] || [120, 200, 220]);

// ===========================================================================
//  Stage 1 — globe intro
// ===========================================================================
function startGlobe() {
  const stations = WORLD.stations.map((s) => ({ lat: s.lat, lng: s.lon }));
  const globe = Globe()(document.getElementById("globe"))
    .backgroundColor("#04070d")
    .globeImageUrl("vendor/earth-dark.jpg")
    .showAtmosphere(true).atmosphereColor("#2dd4ef").atmosphereAltitude(0.18)
    .pointsData(stations).pointLat("lat").pointLng("lng")
    .pointColor(() => "#2dd4ef").pointAltitude(0.01).pointRadius(0.18)
    .ringsData([{ lat: TORONTO.lat, lng: TORONTO.lng }])
    .ringColor(() => (t) => `rgba(45,212,239,${1 - t})`)
    .ringMaxRadius(6).ringPropagationSpeed(3).ringRepeatPeriod(900);

  globe.pointOfView({ lat: 18, lng: -55, altitude: 2.6 }, 0);
  const ctr = globe.controls();
  ctr.autoRotate = true; ctr.autoRotateSpeed = 4.5; ctr.enableZoom = false;

  setTimeout(() => $("globe-cap").classList.add("show"), 1500);
  setTimeout(() => {
    ctr.autoRotate = false;
    globe.pointOfView({ lat: TORONTO.lat, lng: TORONTO.lng, altitude: 0.08 }, 3800);
  }, 6200);
  setTimeout(handoffToMap, 9600);
}

// ===========================================================================
//  Stage 2 — deck.gl Toronto map
// ===========================================================================
function basemapLayer() {
  return new deck.TileLayer({
    id: "carto-dark",
    data: "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
    minZoom: 0, maxZoom: 19, tileSize: 256,
    renderSubLayers: (props) => {
      const t = props.tile;
      let bounds;
      if (t.bbox && t.bbox.west !== undefined) {
        bounds = [t.bbox.west, t.bbox.south, t.bbox.east, t.bbox.north];
      } else if (t.boundingBox) {
        const [[w, s], [e, n]] = t.boundingBox; bounds = [w, s, e, n];
      } else { return null; }
      return new deck.BitmapLayer(props, { data: null, image: props.data, bounds });
    },
  });
}

function tickLayers() {
  const tick = CURRENT || { units: [], moves: [], coverage: {} };
  return [
    basemapLayer(),
    new deck.GeoJsonLayer({
      id: "fsa", data: GEO, stroked: true, filled: true,
      getFillColor: (f) => coverageColor(f.properties.fsa),
      getLineColor: [120, 200, 220, 50], lineWidthMinPixels: 0.5,
      pickable: true, updateTriggers: { getFillColor: RSEQ },
      transitions: { getFillColor: 500 },
    }),
    // candidate posting stations — extruded as 3D "buildings" so they read on
    // the pitched map (flat dots were invisible). cuOpt relocates units onto these.
    new deck.ColumnLayer({
      id: "stations", data: WORLD.stations, pickable: true,
      getPosition: (d) => [d.lon, d.lat],
      diskResolution: 6, radius: 95, extruded: true,
      getElevation: 520, elevationScale: 1,
      getFillColor: [90, 180, 210, 200],
      stroked: true, getLineColor: [150, 230, 255, 160], lineWidthMinPixels: 1,
      material: false,
    }),
    // 911 incident blips — where calls are landing this tick (pulse via glow)
    new deck.ScatterplotLayer({
      id: "incident-glow", data: tick.incidents || [],
      getPosition: (d) => [d.lon, d.lat], getRadius: 600,
      radiusMinPixels: 6, getFillColor: (d) => [...PRIORITY_COLOR(d.priority), 35],
      updateTriggers: { getFillColor: RSEQ, getPosition: RSEQ },
    }),
    new deck.ScatterplotLayer({
      id: "incidents", data: tick.incidents || [], pickable: true,
      getPosition: (d) => [d.lon, d.lat], getRadius: 150,
      radiusMinPixels: 2.5, getFillColor: (d) => PRIORITY_COLOR(d.priority),
      stroked: true, getLineColor: [255, 255, 255, 140], lineWidthMinPixels: 0.6,
      updateTriggers: { getFillColor: RSEQ, getPosition: RSEQ },
    }),
    // dispatch arcs — units responding station -> scene this tick
    new deck.ArcLayer({
      id: "dispatch", data: tick.trips || [],
      getSourcePosition: (d) => d.from, getTargetPosition: (d) => d.to,
      getSourceColor: [45, 212, 239, 90], getTargetColor: [255, 80, 80, 200],
      getWidth: 1.2, getHeight: 0.25, updateTriggers: { data: RSEQ },
    }),
    // units — glide to new posts on relocation (position transition)
    new deck.ScatterplotLayer({
      id: "units-glow", data: tick.units,
      getPosition: (d) => [d.lon, d.lat], getRadius: 520,
      radiusMinPixels: 8, getFillColor: (d) => [...STATUS_COLOR(d.status), 40],
      transitions: { getPosition: 1400 },
      updateTriggers: { getFillColor: RSEQ, getPosition: RSEQ },
    }),
    new deck.ScatterplotLayer({
      id: "units", data: tick.units, pickable: true,
      getPosition: (d) => [d.lon, d.lat], getRadius: 180,
      radiusMinPixels: 3.5, getFillColor: (d) => STATUS_COLOR(d.status),
      stroked: true, getLineColor: [5, 12, 18], lineWidthMinPixels: 1,
      transitions: { getPosition: 1400 },
      updateTriggers: { getFillColor: RSEQ, getPosition: RSEQ },
    }),
    // relocation arcs — cuOpt's repositioning decisions
    new deck.ArcLayer({
      id: "moves", data: tick.moves,
      getSourcePosition: (d) => d.from, getTargetPosition: (d) => d.to,
      getSourceColor: [245, 165, 36], getTargetColor: [45, 212, 239],
      getWidth: 2.5, getHeight: 0.4, updateTriggers: { data: RSEQ },
    }),
  ];
}

function initMap() {
  deckgl = new deck.DeckGL({
    container: "map",
    initialViewState: { longitude: TORONTO.lng, latitude: TORONTO.lat, zoom: 9.8, pitch: 45, bearing: 0 },
    controller: true,
    layers: tickLayers(),
    getTooltip: ({ object }) =>
      object && (object.properties
        ? { html: `<b>${object.properties.fsa}</b><br/>${tooltipFsa(object.properties.fsa)}` }
        : object.status ? { html: `<b>${object.id}</b> · ${object.status}` }
        : object.name ? { html: `<b>${object.name}</b><br/>candidate post` }
        : object.id ? { html: `<b>${object.id}</b>` } : null),
  });
}

function tooltipFsa(fsa) {
  if (WORLD && WORLD.fsa_uncoverable && WORLD.fsa_uncoverable.includes(fsa))
    return "OUT OF NETWORK · no covering station";
  const c = CURRENT && CURRENT.coverage[fsa];
  return c ? `${c.covered ? "covered" : "GAP"} · nearest ${c.nearest_min} min` : "no data";
}

function handoffToMap() {
  initMap();
  $("globe-stage").classList.add("hidden");
  $("map-stage").classList.add("shown");
  if (MODE === "live") { stepLive().then(() => play()); }
  else { CURRENT = TIMELINE.ticks[idx]; renderTick(CURRENT); play(); }
}

// ===========================================================================
//  render a tick (shared by both modes)
// ===========================================================================
function renderTick(tick) {
  CURRENT = tick; RSEQ++;
  if (deckgl) deckgl.setProps({ layers: tickLayers() });

  const cov = tick.covered_pct * 100;
  const avail = tick.units.filter((u) => u.status === "available").length;
  $("t-clock").textContent = `${tick.t} min`;
  $("t-tick").textContent = MODE === "live"
    ? `#${tick.tick_no || "?"} · LIVE`
    : `${idx + 1}/${TIMELINE.ticks.length}`;

  const kcov = $("k-cov");
  kcov.textContent = `${cov.toFixed(1)}%`;
  const tier = cov >= 90 ? "good" : cov >= 60 ? "warn" : "bad";
  kcov.className = "v " + tier;
  const tcov = $("t-cov");
  if (tcov) { tcov.textContent = `${cov.toFixed(1)}%`; tcov.className = "v " + tier; }
  const col = cov >= 90 ? "#25e0a0" : cov >= 60 ? "#f5a524" : "#ff4d4d";
  const bar = $("covbar-i");
  bar.style.width = `${cov}%`; bar.style.background = col; bar.style.boxShadow = `0 0 10px ${col}`;

  const kg = $("k-gaps"); kg.textContent = tick.gaps;
  kg.className = "v " + (tick.gaps <= 5 ? "good" : tick.gaps <= 20 ? "warn" : "bad");
  $("k-avail").textContent = `${avail}/${tick.units.length}`;
  $("k-ontime").textContent = `${(tick.kpi.on_time_pct * 100).toFixed(0)}%`;

  const held = document.body.classList.contains("held");
  $("a-tick").textContent = `T+${tick.t}min · ${tick.moves.length} relocations`
    + (held ? "   ·   ▮ HELD — press ▶ Play to resume live sim" : "");
  $("a-cmd").textContent = tick.command ? `▸ operator: "${tick.command}"` : "";
  $("a-body").textContent = tick.explanation || "—";
  const an = $("a-notes");
  if (an) {
    const notes = (tick.notes || []).join(" · ");
    an.textContent = notes ? `⚠ ${notes}` : "";
    an.style.display = notes ? "block" : "none";
  }

  if (MODE === "live") $("progress-i").style.width = "100%";
  else $("progress-i").style.width = `${((idx + 1) / TIMELINE.ticks.length) * 100}%`;

  renderRoster(tick);
  renderFeed(tick);
  renderSolve(tick);
  renderScene(tick);
}

// ---- side panels ----------------------------------------------------------
function renderRoster(tick) {
  const el = $("roster-list"); if (!el) return;
  const units = (tick.units || []).slice();
  // Gotham-style stack: BUSY units bubble to the top (most-committed first),
  // available units below, alphabetical. Color coding unchanged (busy=amber).
  units.sort((a, b) => {
    const ab = a.status === "available" ? 1 : 0;
    const bb = b.status === "available" ? 1 : 0;
    if (ab !== bb) return ab - bb;                       // busy (0) first
    if (ab === 0) return (b.eta_free || 0) - (a.eta_free || 0); // longest ETA on top
    return String(a.id).localeCompare(String(b.id));
  });
  const busy = units.filter((u) => u.status !== "available").length;
  const rc = $("roster-count");
  if (rc) rc.textContent = `${busy} ON CALL`;
  const ra = $("roster-avail");
  if (ra) ra.textContent = `${units.length - busy} READY`;

  el.innerHTML = units.map((u) => {
    const on = u.status !== "available";
    const badge = on ? "ON CALL" : "READY";
    const eta = u.eta_free != null ? `${u.eta_free}m` : "—";
    const sub = on ? `responding · frees in ${eta}` : "posted · standing by";
    const post = u.post || u.station_id || "—";
    return `<div class="unit-card ${on ? "busy" : "available"}">
      <div class="uc-ic">${on ? "▲" : "◆"}</div>
      <div class="uc-main">
        <div class="uc-top"><span class="uc-id">${u.id}</span><span class="uc-badge">${badge}</span></div>
        <div class="uc-sub">${sub}</div>
        <div class="uc-q"><span class="uc-post">⌖ ${post}</span><span class="uc-tag">TPS//EMS</span></div>
      </div>
    </div>`;
  }).join("");
}

function renderFeed(tick) {
  const el = $("feed-list"); if (!el) return;
  // prepend this tick's incidents (newest first); cap the buffer
  for (const c of (tick.incidents || [])) {
    const cls = c.served ? "" : "missed";
    const right = c.served ? `${c.unit} · ${c.resp}m` : "⚠ NO UNIT";
    FEED.unshift(`<div class="feed-row ${cls}"><span class="pr ${c.priority}">${c.priority || "—"}</span>`
      + `<span class="meta">${c.type} · ${c.zone} · ${right}</span></div>`);
  }
  FEED = FEED.slice(0, 60);
  el.innerHTML = FEED.join("");
  const fr = $("feed-rate"); if (fr) fr.textContent = `${tick.calls_this_tick || 0}/tick`;
}

function renderSolve(tick) {
  const el = $("solve-body"); if (!el) return;
  const s = tick.solve;
  if (s) {
    el.className = "";
    el.innerHTML = `Relocation MILP solved on <b>GB10</b><br/>`
      + `${s.status} · <b>${s.solve_time_ms} ms</b> · ${s.n_moves} move(s)`;
  } else if (tick.scene === "SURGE") {
    el.className = "solving";
    el.textContent = "demand surge — coverage degrading…";
  } else {
    el.className = "idle";
    el.textContent = "standing by · GB10";
  }
}

function renderScene(tick) {
  const sc = tick.scene || "NORMAL";
  document.body.classList.remove("scene-NORMAL", "scene-SURGE", "scene-SOLVE", "scene-RECOVERED", "scene-LIVE");
  document.body.classList.add("scene-" + sc);
  const spark = $("spark");
  if (spark) spark.classList.toggle("show", sc === "RECOVERED" || sc === "SOLVE");
}

// ===========================================================================
//  stepping — static vs live
// ===========================================================================
function stepStatic() {
  idx = (idx + 1) % TIMELINE.ticks.length;
  if (idx === 0) FEED = [];                 // restart feed on loop
  renderTick(TIMELINE.ticks[idx]);
}

async function stepLive() {
  if (busy) return;
  busy = true;
  const cmd = pendingCommand; pendingCommand = null;
  if (cmd) $("a-body").textContent = "◢ agent thinking…";
  try {
    const r = await fetch("/api/step", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: cmd }),
    });
    const tick = await r.json();
    if (tick.error) { $("a-body").textContent = "backend error: " + tick.error; }
    else renderTick(tick);
  } catch (e) {
    $("a-body").textContent = "backend unreachable: " + e.message;
  } finally { busy = false; }
}

async function runCommandStep() {
  // wait for any in-flight tick to finish, then run exactly one command step
  while (busy) await new Promise((r) => setTimeout(r, 120));
  await stepLive();   // stays paused afterwards so the result is held on screen
}

// ===========================================================================
//  DISPATCH PIPELINE — audio → Parakeet → subtitles → Nemotron → cuOpt
// ===========================================================================
let CLIPS = [];
const PSUB = { asr: "transcribe dispatch audio",
               nemotron: "parse intent → constraints",
               cuopt: "relocation MILP · GPU" };

function stage(step, state, sub) {
  const el = document.querySelector(`#pipe-steps .pstep[data-step="${step}"]`);
  if (!el) return;
  el.classList.remove("active", "done");
  if (state) el.classList.add(state);
  const ps = el.querySelector(".psub");
  if (ps) ps.textContent = sub != null ? sub : PSUB[step];
}
function resetPipeline() { ["asr", "nemotron", "cuopt"].forEach((s) => stage(s, null)); }

function showSubtitle() {
  const s = $("subtitle"); if (!s) return;
  $("sub-text").innerHTML = ""; s.classList.add("show");
}

// reveal the transcript word-by-word, paced to audio playback (live-caption
// illusion: Parakeet returns the whole transcript, we stream it over the clip)
function playWithCaptions(audio, transcript) {
  return new Promise((resolve) => {
    const words = transcript.split(/\s+/).filter(Boolean);
    const el = $("sub-text");
    if (!words.length) { el.textContent = "(no speech detected)"; return resolve(); }
    let done = false, fb = false;
    const render = (frac) => {
      const n = Math.max(1, Math.round(words.length * Math.min(1, frac)));
      const shown = words.slice(0, n).join(" ");
      const pend = words.slice(n).join(" ");
      el.innerHTML = shown + (pend ? ` <span class="pend">${pend}</span>` : "")
        + ` <span class="cursor">▌</span>`;
    };
    const finish = () => { if (done) return; done = true; el.textContent = transcript; resolve(); };
    const fallbackType = () => {
      if (fb || done) return; fb = true;
      let i = 0; const per = Math.max(70, Math.min(170, 2600 / words.length));
      const iv = setInterval(() => { render(++i / words.length);
        if (i >= words.length) { clearInterval(iv); finish(); } }, per);
    };
    audio.addEventListener("timeupdate", () => {
      if (audio.duration) render(audio.currentTime / audio.duration);
    });
    audio.addEventListener("ended", finish);
    audio.addEventListener("error", fallbackType);
    const p = audio.play();
    if (p && p.catch) p.catch(fallbackType);
    setTimeout(() => { if (!done && (!audio.duration || audio.paused)) fallbackType(); }, 1300);
  });
}

async function runDispatch(clip) {
  if (busy) return;
  busy = true;
  pause();
  document.body.classList.add("held");
  document.querySelectorAll(".chip").forEach((c) => c.classList.add("busy"));
  resetPipeline();
  $("a-cmd").textContent = `▸ radio: "${clip.label}"`;
  $("a-body").textContent = "◢ receiving dispatch…";
  showSubtitle();
  const sub = $("subtitle");
  try {
    // 1) Parakeet ASR ----------------------------------------------------
    stage("asr", "active");
    sub.classList.add("playing");
    const audio = new Audio(clip.audio_url);
    const t0 = performance.now();
    const dres = await fetch("/api/dispatch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_id: clip.id }),
    }).then((r) => r.json());
    if (dres.error) { stage("asr", null, dres.error); throw new Error(dres.error); }
    const transcript = (dres.transcript || "").trim();
    const asrMs = Math.round(performance.now() - t0);
    stage("asr", "done", `${transcript.split(/\s+/).length} words · ${asrMs} ms`);

    // 2) play audio + live captions -------------------------------------
    await playWithCaptions(audio, transcript);
    sub.classList.remove("playing");

    // 3) Nemotron parse + cuOpt solve (one backend call) ----------------
    stage("nemotron", "active");
    const cu = setTimeout(() => {
      stage("nemotron", "done", "intent parsed");
      stage("cuopt", "active");
    }, 650);
    const tick = await fetch("/api/step", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: transcript }),
    }).then((r) => r.json());
    clearTimeout(cu);
    if (tick.error) { stage("cuopt", null, "error"); $("a-body").textContent = "backend error: " + tick.error; return; }
    stage("nemotron", "done", "intent parsed");
    const sv = tick.solve;
    stage("cuopt", "done", sv ? `${sv.status} · ${sv.solve_time_ms} ms · ${sv.n_moves} move(s)` : "solved");
    renderTick(tick);   // map + LLM explanation + moves
  } catch (e) {
    sub.classList.remove("playing");
    $("a-body").textContent = "dispatch failed: " + e.message;
  } finally {
    busy = false;
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("busy"));
  }
}

function renderChips() {
  const el = $("dispatch-chips"); if (!el) return;
  el.innerHTML = "";
  CLIPS.forEach((c) => {
    const b = document.createElement("span");
    b.className = "chip"; b.textContent = c.label;
    b.onclick = () => runDispatch(c);
    el.appendChild(b);
  });
}

function play() {
  playing = true; $("btn-pause").textContent = "❚❚ Pause";
  document.body.classList.remove("held");
  clearTimeout(timer); clearInterval(timer);
  if (MODE === "live") {
    const loop = async () => {
      if (!playing) return;
      await stepLive();
      if (playing) timer = setTimeout(loop, TICK_MS);
    };
    loop();
  } else {
    timer = setInterval(stepStatic, TICK_MS);
  }
}
function pause() {
  playing = false; $("btn-pause").textContent = "▶ Play";
  clearTimeout(timer); clearInterval(timer);
}

// ===========================================================================
//  boot
// ===========================================================================
async function detectBackend() {
  try {
    const r = await fetch("/api/health", { cache: "no-store" });
    if (!r.ok) return false;
    const h = await r.json();
    return !!h.live;
  } catch { return false; }
}

async function main() {
  GEO = await fetch("data/toronto_fsa.geojson").then((r) => r.json());

  if (await detectBackend()) {
    MODE = "live";
    WORLD = await fetch("/api/world").then((r) => r.json());
    document.body.classList.add("live");
    CLIPS = await fetch("/api/clips").then((r) => r.json()).catch(() => []);
    renderChips();
  } else {
    MODE = "static";
    [WORLD, TIMELINE] = await Promise.all([
      fetch("data/world.json").then((r) => r.json()),
      fetch("data/timeline.json").then((r) => r.json()),
    ]);
  }
  MAXDEMAND = Math.max(...Object.values(WORLD.fsa_demand));
  $("mode-tag").textContent = MODE === "live" ? "LIVE · cuOpt + Nemotron" : "REPLAY";

  $("btn-pause").onclick = () => (playing ? pause() : play());
  $("btn-replay").onclick = () => {
    pause(); idx = 0; FEED = [];
    $("map-stage").classList.remove("shown");
    $("globe-stage").classList.remove("hidden");
    $("globe").innerHTML = "";
    startGlobe();
  };

  // live command box
  const send = () => {
    const v = $("cmd-in").value.trim();
    if (!v) return;
    pause();                                  // hold the live sim
    document.body.classList.add("held");
    pendingCommand = v;
    $("a-cmd").textContent = `▸ operator: "${v}"`;
    $("a-body").textContent = "◢ agent thinking…";
    $("cmd-in").value = "";
    runCommandStep();                          // run it, then stay held
  };
  if ($("cmd-send")) {
    $("cmd-send").onclick = send;
    $("cmd-in").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
  }

  setTimeout(() => { $("boot").classList.add("gone"); startGlobe(); }, 1400);
}
main();
