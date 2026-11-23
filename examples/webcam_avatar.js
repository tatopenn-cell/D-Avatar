/* D-Avatar Studio — Live Input. Webcam + Face Mesh + Lip-sync + disegno + mani. */

const state = {
  cameraOn: false,
  cleanMode: false,
  video: null,
  stream: null,
  cameraAvailable: false,
  faceLandmarks: null,
  handLandmarks: null,
  currentLayer: 'head',
  layers: {},
  brushSize: 8,
  brushColor: '#ffffff',
  isDrawing: false,
  lastTime: 0,
  fps: 0,
  frameCount: 0,
  fpsTime: 0,
  hands: null,
  handsReady: false,
  faceMesh: null,
  faceReady: false,
  syntheticTime: 0,
  mouseHand: { x: null, y: null, active: false },
  // Lip-sync (microfono)
  mic: null,
  micAnalyser: null,
  micData: null,
  micReady: false,
  mouthOpenByAudio: 0,
  // Parametri API Dense-Armor (opzionali)
  apiParams: {},
  lastApiSend: 0,
};

const els = {};
function $(id) { return document.getElementById(id); }

function init() {
  const required = ['main-canvas','draw-canvas','webcam-video','pip-cam','badge-face','badge-hand','badge-fps','badge-mic','crosshair','avatar-svg'];
  for (const id of required) {
    if (!document.getElementById(id)) {
      const msg = document.createElement('div');
      msg.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#EF4444;color:#fff;padding:20px;border-radius:8px;font-family:sans-serif;z-index:9999;';
      msg.textContent = 'Elemento mancante: #' + id + '. Ricarica la pagina.';
      document.body.appendChild(msg);
      return;
    }
  }
  els.canvas = $('main-canvas');
  els.ctx = els.canvas.getContext('2d', { willReadFrequently: true });
  els.draw = $('draw-canvas');
  els.dctx = els.draw.getContext('2d', { willReadFrequently: true });
  els.video = $('webcam-video');
  els.pip = $('pip-cam');
  els.pipCtx = els.pip.getContext('2d');
  els.faceBadge = $('badge-face');
  els.handBadge = $('badge-hand');
  els.fpsBadge = $('badge-fps');
  els.micBadge = $('badge-mic');
  els.crosshair = $('crosshair');
  state.dctx = els.dctx;

  // Riferimenti SVG livelli avatar
  state.svg = {
    head: $('#svg-head'),
    neck: $('#svg-neck'),
    hair: $('#svg-hair'),
    brows: $('#svg-brows'),
    browL: $('#brow-left'),
    browR: $('#brow-right'),
    eyes: $('#svg-eyes'),
    eyeL: $('#eye-left'),
    eyeR: $('#eye-right'),
    pupilL: $('#pupil-left'),
    pupilR: $('#pupil-right'),
    nose: $('#svg-nose'),
    mouth: $('#svg-mouth'),
    mouthLip: $('#mouth-lip'),
    mouthInside: $('#mouth-inside'),
    outline: $('#svg-outline'),
  };

  initLayers();
  initEvents();
  resizeCanvases();
  window.addEventListener('resize', resizeCanvases);

  initMic();

  startWebcam().then(() => {
    requestAnimationFrame(tick);
  }).catch(err => {
    console.warn('Webcam non disponibile, uso placeholder sintetico:', err);
    state.cameraAvailable = false;
    $('btn-camera').textContent = 'ATTIVA CAMERA';
    $('btn-camera').style.background = 'transparent';
    $('btn-camera').style.color = 'var(--text-primary)';
    els.faceBadge.textContent = 'CAM: NEGATA';
    els.faceBadge.className = 'pill-small pill-red';
    els.handBadge.textContent = 'HAND: --';
    els.handBadge.className = 'pill-small pill-red';
    state.syntheticTime = 0;
    requestAnimationFrame(tick);
  });

  initHands();
  initFaceMesh();
}

function initLayers() {
  state.layers = {};
  const names = ['head','eyes-open','eyes-closed','mouth-open','mouth-closed','hand-default','hand-horns'];
  names.forEach(name => {
    const c = document.createElement('canvas');
    c.width = els.draw.width;
    c.height = els.draw.height;
    state.layers[name] = { canvas: c, ctx: c.getContext('2d') };
  });
  drawDefaultFace();
}

function drawDefaultFace() {
  const l = state.layers;
  const c = l.head.ctx;
  c.fillStyle = '#FDBA74';
  c.beginPath(); c.arc(150, 150, 110, 0, Math.PI * 2); c.fill();
  c.strokeStyle = '#111827'; c.lineWidth = 4; c.stroke();
  c.fillStyle = '#FFFFFF';
  c.beginPath(); c.arc(115, 130, 22, 0, Math.PI * 2); c.fill();
  c.beginPath(); c.arc(185, 130, 22, 0, Math.PI * 2); c.fill();
  c.fillStyle = '#06B6D4';
  c.beginPath(); c.arc(115, 130, 10, 0, Math.PI * 2); c.fill();
  c.beginPath(); c.arc(185, 130, 10, 0, Math.PI * 2); c.fill();
  c.fillStyle = '#F472B6';
  c.beginPath(); c.arc(150, 180, 25, 0, Math.PI); c.fill();

  const ec = l['eyes-closed'].ctx;
  ec.fillStyle = '#FFFFFF';
  ec.beginPath(); ec.arc(115, 130, 22, 0, Math.PI * 2); ec.fill();
  ec.beginPath(); ec.arc(185, 130, 22, 0, Math.PI * 2); ec.fill();
  ec.strokeStyle = '#111827'; ec.lineWidth = 4;
  ec.beginPath(); ec.moveTo(93, 130); ec.lineTo(137, 130); ec.stroke();
  ec.beginPath(); ec.moveTo(163, 130); ec.lineTo(207, 130); ec.stroke();

  const mo = l['mouth-open'].ctx;
  mo.fillStyle = '#7F1D1D';
  mo.beginPath(); mo.arc(150, 185, 28, 0, Math.PI * 2); mo.fill();
  mo.fillStyle = '#FFFFFF';
  mo.fillRect(138, 175, 10, 12);
  mo.fillRect(152, 175, 10, 12);

  const mc = l['mouth-closed'].ctx;
  mc.fillStyle = '#F472B6';
  mc.beginPath(); mc.arc(150, 180, 25, 0, Math.PI); mc.fill();

  const hd = l['hand-default'].ctx;
  hd.fillStyle = '#FDBA74';
  hd.beginPath(); hd.arc(150, 150, 24, 0, Math.PI * 2); hd.fill();
  hd.strokeStyle = '#111827'; hd.lineWidth = 3; hd.stroke();

  const hh = l['hand-horns'].ctx;
  hh.fillStyle = '#FDBA74';
  hh.beginPath(); hh.arc(150, 150, 24, 0, Math.PI * 2); hh.fill();
  hh.strokeStyle = '#EF4444'; hh.lineWidth = 6;
  hh.beginPath(); hh.moveTo(150, 150); hh.lineTo(150, 90); hh.stroke();
  hh.beginPath(); hh.moveTo(150, 150); hh.lineTo(190, 90); hh.stroke();
}

function resizeCanvases() {
  const dpr = window.devicePixelRatio || 1;
  const w = 640, h = 480;
  els.canvas.width = w * dpr; els.canvas.height = h * dpr;
  els.canvas.style.width = w + 'px'; els.canvas.style.height = h + 'px';
  els.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const saved = {
    strokeStyle: state.dctx?.strokeStyle ?? '#ffffff',
    lineWidth: state.dctx?.lineWidth ?? 8,
    lineCap: state.dctx?.lineCap ?? 'round',
    lineJoin: state.dctx?.lineJoin ?? 'round',
  };

  els.draw.width = 300;
  els.draw.height = 300;
  els.dctx.setTransform(1, 0, 0, 1, 0, 0);

  state.dctx.strokeStyle = saved.strokeStyle;
  state.dctx.lineWidth = saved.lineWidth;
  state.dctx.lineCap = saved.lineCap;
  state.dctx.lineJoin = saved.lineJoin;
}

/* ---------- Webcam ---------- */
async function startWebcam() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    els.video.srcObject = state.stream;
    await new Promise(resolve => { els.video.onloadedmetadata = resolve; });
    els.video.play();
    state.cameraOn = true;
    state.cameraAvailable = true;
    $('webcam-toggle').checked = true;
    $('btn-camera').textContent = 'SPEGNI CAMERA';
    $('btn-camera').style.background = 'var(--accent-primary)';
    $('btn-camera').style.color = '#fff';
    els.faceBadge.textContent = 'CAM: OK';
    els.faceBadge.className = 'pill-small pill-green';
  } catch (e) {
    state.cameraOn = false;
    state.cameraAvailable = false;
    $('btn-camera').textContent = 'ATTIVA CAMERA';
    $('btn-camera').style.background = 'transparent';
    $('btn-camera').style.color = 'var(--text-primary)';
    els.faceBadge.textContent = 'CAM: NEGATA';
    els.faceBadge.className = 'pill-small pill-red';
    throw e;
  }
}

/* ---------- Microfono (Lip-sync) ---------- */
async function initMic() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) throw new Error('AudioContext non supportato');
    state.mic = new AC();
    const source = state.mic.createMediaStreamSource(stream);
    state.micAnalyser = state.mic.createAnalyser();
    state.micAnalyser.fftSize = 256;
    state.micData = new Uint8Array(state.micAnalyser.frequencyBinCount);
    source.connect(state.micAnalyser);
    state.micReady = true;
    els.micBadge.textContent = 'MIC: OK';
    els.micBadge.className = 'pill-small pill-green';
  } catch (e) {
    console.warn('Microfono non disponibile:', e);
    state.micReady = false;
    els.micBadge.textContent = 'MIC: NEGATA';
    els.micBadge.className = 'pill-small pill-red';
  }
}

function getMicVolume() {
  if (!state.micReady || !state.micAnalyser) return 0;
  state.micAnalyser.getByteFrequencyData(state.micData);
  let sum = 0;
  for (let i = 0; i < state.micData.length; i++) sum += state.micData[i];
  const avg = sum / state.micData.length;
  // Normalizza 0-255 -> 0-1 circa
  return Math.min(1, avg / 120);
}

/* ---------- MediaPipe Hands ---------- */
function loadHandsScript() {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1650995939/hands.js';
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('CDN bloccato o non raggiungibile'));
    document.head.appendChild(script);
  });
}

async function initHands() {
  try {
    if (!window.Hands) await loadHandsScript();
    const Hands = window.Hands;
    if (!Hands) {
      state.hands = null;
      state.handsReady = false;
      els.handBadge.textContent = 'HAND: --';
      els.handBadge.className = 'pill-small pill-red';
      return;
    }
    state.hands = new Hands({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1650995939/${f}`,
    });
    state.hands.setOptions({
      maxNumHands: 2,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5,
    });
    state.hands.onResults(onHandsResults);
    state.handsReady = true;
    els.handBadge.textContent = 'HAND: PRONTO';
    els.handBadge.className = 'pill-small pill-green';
  } catch (e) {
    console.warn('Impossibile inizializzare MediaPipe Hands:', e);
    state.hands = null;
    state.handsReady = false;
    els.handBadge.textContent = 'HAND: ERRORE';
    els.handBadge.className = 'pill-small pill-red';
  }
}

async function retryHands() {
  state.hands = null;
  state.handsReady = false;
  els.handBadge.textContent = 'HAND: CARICAMENTO...';
  els.handBadge.className = 'pill-small pill-amber';
  await initHands();
}

function onHandsResults(results) {
  state.handLandmarks = results;
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    els.handBadge.textContent = 'HAND: OK';
    els.handBadge.className = 'pill-small pill-green';
  } else {
    els.handBadge.textContent = 'HAND: --';
    els.handBadge.className = 'pill-small pill-red';
  }
}

/* ---------- MediaPipe Face Mesh ---------- */
function loadFaceMeshScript() {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1650995939/face_mesh.js';
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('CDN bloccato o non raggiungibile'));
    document.head.appendChild(script);
  });
}

async function initFaceMesh() {
  try {
    if (!window.FaceMesh) await loadFaceMeshScript();
    const FaceMesh = window.FaceMesh;
    if (!FaceMesh) {
      state.faceMesh = null;
      state.faceReady = false;
      els.faceBadge.textContent = 'FACE: --';
      els.faceBadge.className = 'pill-small pill-red';
      return;
    }
    state.faceMesh = new FaceMesh({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1650995939/${f}`,
    });
    state.faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    state.faceMesh.onResults(onFaceResults);
    state.faceReady = true;
    els.faceBadge.textContent = 'FACE: PRONTO';
    els.faceBadge.className = 'pill-small pill-green';
  } catch (e) {
    console.warn('Impossibile inizializzare MediaPipe Face Mesh:', e);
    state.faceMesh = null;
    state.faceReady = false;
    els.faceBadge.textContent = 'FACE: ERRORE';
    els.faceBadge.className = 'pill-small pill-red';
  }
}

function onFaceResults(results) {
  state.faceLandmarks = results;
  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    els.faceBadge.textContent = 'FACE: OK';
    els.faceBadge.className = 'pill-small pill-green';
  } else {
    els.faceBadge.textContent = 'FACE: --';
    els.faceBadge.className = 'pill-small pill-red';
  }
}

/* ---------- Eventi ---------- */
function initEvents() {
  $('btn-camera').addEventListener('click', async () => {
    if (state.cameraOn && state.stream) {
      state.cameraOn = false;
      state.stream.getTracks().forEach(t => t.stop());
      state.stream = null;
      $('btn-camera').textContent = 'ATTIVA CAMERA';
      $('btn-camera').style.background = 'transparent';
      $('btn-camera').style.color = 'var(--text-primary)';
      els.faceBadge.textContent = 'CAM: SPENTA';
      els.faceBadge.className = 'pill-small pill-red';
      return;
    }
    try {
      await startWebcam();
      $('btn-camera').textContent = 'SPEGNI CAMERA';
      $('btn-camera').style.background = 'var(--accent-primary)';
      $('btn-camera').style.color = '#fff';
      els.faceBadge.textContent = 'CAM: OK';
      els.faceBadge.className = 'pill-small pill-green';
      $('camera-error').style.display = 'none';
    } catch (e) {
      $('btn-camera').textContent = 'ATTIVA CAMERA';
      els.faceBadge.textContent = 'CAM: NEGATA';
      els.faceBadge.className = 'pill-small pill-red';
      showCameraError(e.message);
    }
  });

  $('webcam-toggle').addEventListener('change', (e) => {
    if (e.target.checked && !state.cameraOn) $('btn-camera').click();
    else if (!e.target.checked && state.cameraOn) $('btn-camera').click();
  });

  $('btn-test-cam').addEventListener('click', async () => {
    $('camera-error').style.display = 'none';
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      stream.getTracks().forEach(t => t.stop());
      showCameraError('Webcam FUNZIONA. Il permesso è stato concesso.', true);
    } catch (e) {
      showCameraError(e.message, false);
    }
  });

  $('btn-clean').addEventListener('click', () => {
    state.cleanMode = !state.cleanMode;
    applyCleanMode();
  });

  els.draw.addEventListener('mousedown', (e) => {
    state.isDrawing = true;
    const rect = els.draw.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    state.dctx.beginPath();
    state.dctx.moveTo(x, y);
    updateCrosshair(e.clientX, e.clientY, true);
  });
  els.draw.addEventListener('mousemove', (e) => {
    updateCrosshair(e.clientX, e.clientY, state.isDrawing);
    if (!state.isDrawing) return;
    const rect = els.draw.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    state.dctx.lineTo(x, y);
    state.dctx.stroke();
  });
  els.draw.addEventListener('mouseup', () => { state.isDrawing = false; updateCrosshair(0, 0, false); });
  els.draw.addEventListener('mouseleave', () => { state.isDrawing = false; updateCrosshair(0, 0, false); });

  state.dctx.strokeStyle = state.brushColor;
  state.dctx.lineWidth = state.brushSize;
  state.dctx.lineCap = 'round';
  state.dctx.lineJoin = 'round';

  $('brush-size').addEventListener('input', (e) => {
    state.brushSize = parseInt(e.target.value);
    state.dctx.lineWidth = state.brushSize;
  });
  $('color-picker').addEventListener('input', (e) => {
    state.brushColor = e.target.value;
    state.dctx.strokeStyle = state.brushColor;
    updateSwatches(null);
  });
  document.querySelectorAll('.c-swatch').forEach(sw => {
    sw.addEventListener('click', () => {
      state.brushColor = sw.dataset.color;
      state.dctx.strokeStyle = state.brushColor;
      $('color-picker').value = state.brushColor;
      updateSwatches(sw);
    });
  });
  $('btn-undo').addEventListener('click', () => { clearLayer(state.currentLayer); });
  $('btn-clear').addEventListener('click', () => { clearLayer(state.currentLayer); });
  $('btn-load-png').addEventListener('click', () => { $('png-file').click(); });
  $('png-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => { state.dctx.drawImage(img, 0, 0, els.draw.width, els.draw.height); };
    img.src = URL.createObjectURL(file);
  });
  document.querySelectorAll('#layer-tabs .tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('#layer-tabs .tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.currentLayer = tab.dataset.layer;
      updateSubtabs();
    });
  });
  document.querySelectorAll('#subtabs .tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('#subtabs .tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
    });
  });
  $('save-png').addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = 'avatar-' + state.currentLayer + '-' + Date.now() + '.png';
    link.href = els.draw.toDataURL('image/png');
    link.click();
  });
  $('btn-apply').addEventListener('click', () => { requestAnimationFrame(tick); });
  $('btn-export').addEventListener('click', () => {
    const data = {};
    Object.keys(state.layers).forEach(k => { data[k] = state.layers[k].canvas.toDataURL('image/png'); });
    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'd-avatar-character.json'; a.click();
    URL.revokeObjectURL(url);
  });
  $('btn-import').addEventListener('click', () => { $('json-import').click(); });
  $('json-import').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target.result);
        Object.keys(data).forEach(k => {
          if (state.layers[k] && data[k]) {
            const img = new Image();
            img.onload = () => state.layers[k].ctx.drawImage(img, 0, 0, state.layers[k].canvas.width, state.layers[k].canvas.height);
            img.src = data[k];
          }
        });
      } catch (err) { alert('File JSON non valido'); }
    };
    reader.readAsText(file);
  });
  $('sw-black').addEventListener('click', () => setBg('black'));
  $('sw-green').addEventListener('click', () => setBg('green'));
  $('sw-trans').addEventListener('click', () => setBg('trans'));

  $('camera-error-close')?.addEventListener('click', () => { $('camera-error').style.display = 'none'; });
  $('btn-retry-hands').addEventListener('click', async () => { await retryHands(); });

  $('btn-pipeline').addEventListener('click', async () => {
    try {
      const r = await fetch('http://127.0.0.1:5000/pipeline/build', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ period_s: 0.05, max_step: 0.1, max_step_deg: 5.0 })
      });
      const data = await r.json();
      showCameraError('Pipeline costruita: ' + JSON.stringify(data), true);
    } catch (e) {
      showCameraError('Errore pipeline: ' + e.message, false);
    }
  });

  // Mano sintetica mouse fallback
  els.canvas.addEventListener('mousemove', (e) => {
    const rect = els.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    state.mouseHand.x = (e.clientX - rect.left) * dpr;
    state.mouseHand.y = (e.clientY - rect.top) * dpr;
    state.mouseHand.active = true;
  });
  els.canvas.addEventListener('mouseleave', () => { state.mouseHand.active = false; });
}

function updateCrosshair(x, y, visible) {
  const cr = els.crosshair;
  if (visible) { cr.style.left = x + 'px'; cr.style.top = y + 'px'; cr.style.display = 'block'; }
  else { cr.style.display = 'none'; }
}

function updateSwatches(activeEl) {
  document.querySelectorAll('.c-swatch').forEach(s => s.classList.remove('active'));
  if (activeEl) activeEl.classList.add('active');
}

function clearLayer(name) {
  const l = state.layers[name];
  if (l) {
    l.ctx.clearRect(0, 0, l.canvas.width, l.canvas.height);
    if (name === 'head') drawDefaultFace();
  }
}

function updateSubtabs() {
  const subs = $('subtabs');
  const hasSub = ['eyes','mouth','hand'].includes(state.currentLayer);
  subs.style.display = hasSub ? 'flex' : 'none';
}

function showCameraError(msg, ok) {
  const box = $('camera-error');
  const m = $('camera-error-msg');
  const h = $('camera-error-hint');
  m.textContent = msg;
  box.style.display = 'block';
  if (ok) {
    box.style.background = 'rgba(16,185,129,0.95)';
    h.textContent = 'La webcam è accessibile. Clicca ATTIVA CAMERA per usarla nell\'app.';
  } else {
    box.style.background = 'rgba(239,68,68,0.95)';
    h.textContent = 'Cosa fare:\n1. Clicca l\'icona lucchetto 🔒 nella barra indirizzi del browser e imposta la fotocamera su Consenti.\n2. Ricarica la pagina.\n3. Usa http://127.0.0.1:8000/webcam_avatar.html (NON aprire il file con doppio clic).\n4. Chiudi altre app che usano la webcam (Zoom, Teams, OBS, Chrome con altre tab).';
  }
}

function applyCleanMode() {
  const header = document.querySelector('header');
  const panel = document.querySelector('.panel-right');
  const pad = document.querySelector('.pad-wrap');
  const wrap = document.querySelector('#canvas-wrap');
  const hud = document.querySelector('.hud');
  if (state.cleanMode) {
    header.style.display = 'none';
    panel.style.display = 'none';
    pad.style.display = 'none';
    wrap.style.border = 'none';
    hud.style.display = 'none';
    document.querySelector('.stage-area').style.background = '#000';
  } else {
    header.style.display = '';
    panel.style.display = '';
    pad.style.display = '';
    wrap.style.border = '1px solid var(--border-subtle)';
    hud.style.display = '';
    document.querySelector('.stage-area').style.background = '#0a0a12';
  }
}

function setBg(mode) {
  const stage = document.querySelector('.stage-area');
  const body = document.body;
  document.querySelectorAll('.sw').forEach(s => s.classList.remove('active'));
  if (mode === 'black') {
    $('sw-black').classList.add('active');
    body.style.background = '#000000';
    stage.style.background = '#000000';
  } else if (mode === 'green') {
    $('sw-green').classList.add('active');
    body.style.background = '#00FF00';
    stage.style.background = '#00FF00';
  } else {
    $('sw-trans').classList.add('active');
    body.style.background = '#0D0F12';
    stage.style.background = '#0a0a12';
  }
}

/* ---------- Avatar SVG (livelli separati, niente canvas primitivi) ---------- */
function applyFaceToSVG(lm) {
  if (!lm || !lm.multiFaceLandmarks || lm.multiFaceLandmarks.length === 0) return;
  const face = lm.multiFaceLandmarks[0];

  // Indici Face Mesh (approssimativi, validi per la mesh standard):
  // 13 = labbro superiore centro, 14 = labbro inferiore centro (apertura bocca)
  // 33 = angolo occhio sx, 263 = angolo occhio dx (giro testa)
  // 1 = punta naso (riferimento testa), 10 = fronte, 152 = mento
  const nose = face[1];
  const chin = face[152];
  const forehead = face[10];
  const lipUp = face[13];
  const lipDown = face[14];
  const eyeL = face[33];
  const eyeR = face[263];
  const browL = face[105]; // sopracciglia sx
  const browR = face[334]; // sopracciglia dx

  // --- Testa: scala e rotazione basate su distanza naso-mento e fronte ---
  const headHeight = chin.y - forehead.y;
  const scale = 1 + (headHeight - 0.2) * 1.2; // normalizzazione approssimativa
  const sx = Math.max(0.7, Math.min(1.4, scale));
  const sy = Math.max(0.7, Math.min(1.4, scale));

  // Rotazione testa (yaw/pitch) da posizione naso rispetto al centro della faccia
  const faceCx = (eyeL.x + eyeR.x) / 2;
  const faceCy = (eyeL.y + eyeR.y + nose.y + chin.y) / 4;
  const dx = (nose.x - faceCx) * 120; // gradi max ~±60
  const dy = (nose.y - faceCy) * 100;

  const s = state.svg;
  s.head.setAttribute('transform', `translate(320,240) scale(${sx},${sy}) rotate(${-dx * 0.6},0,0)`);
  s.neck.setAttribute('transform', `translate(320,240) scale(${sx * 0.8},${sy * 0.8}) rotate(${-dy * 0.3},0,0)`);
  s.hair.setAttribute('transform', `translate(320,240) scale(${sx * 1.05},${sy * 1.05})`);
  s.outline.setAttribute('transform', `translate(320,240) scale(${sx * 1.03},${sy * 1.03}) rotate(${-dx * 0.5},0,0)`);

  // --- Sopracciglia: alzate/abbassate in base a posizione landmarks ---
  const browLift = (browL.y + browR.y) / 2 - (forehead.y + 0.02);
  const browRot = browLift * 40;
  s.browL.setAttribute('transform', `translate(-55 + ${-dx * 0.3},-35 + ${browRot}) rotate(${browRot * 0.3})`);
  s.browR.setAttribute('transform', `translate(15 + ${-dx * 0.3},-35 + ${browRot}) rotate(${browRot * 0.3})`);

  // --- Occhi: apertura in base alla distanza verticale palpebra ---
  // Usiamo landmarks interni occhi (159 sx, 386 dx per palpebra superiore)
  const eyeTopL = face[159];
  const eyeBotL = face[145];
  const eyeTopR = face[386];
  const eyeBotR = face[374];
  const openL = Math.max(0, (eyeBotL.y - eyeTopL.y) / 0.015); // ~0-1
  const openR = Math.max(0, (eyeBotR.y - eyeTopR.y) / 0.015);
  const scaleEyeL = Math.min(1, Math.max(0.1, openL * 2));
  const scaleEyeR = Math.min(1, Math.max(0.1, openR * 2));

  s.eyeL.setAttribute('transform', `translate(-58 + ${-dx * 0.2},-25 + ${dy * 0.2}) scale(1,${scaleEyeL})`);
  s.eyeR.setAttribute('transform', `translate(58 + ${-dx * 0.2},-25 + ${dy * 0.2}) scale(1,${scaleEyeR})`);

  // Pupille seguono sguardo (dx/dy naso)
  const px = (nose.x - faceCx) * 60;
  const py = (nose.y - faceCy) * 40;
  s.pupilL.setAttribute('cx', px * 0.5);
  s.pupilL.setAttribute('cy', py * 0.5);
  s.pupilR.setAttribute('cx', px * 0.5);
  s.pupilR.setAttribute('cy', py * 0.5);

  // --- Bocca: apertura in base a distanza labbra + audio lip-sync ---
  const mouthDist = Math.max(0, lipDown.y - lipUp.y);
  const mouthFromFace = Math.min(1, mouthDist / 0.04);
  const mouthFromAudio = state.mouthOpenByAudio;
  const mouthOpen = Math.max(mouthFromFace, mouthFromAudio);

  // Apertura bocca SVG: path si deforma verso il basso
  const openH = mouthOpen * 20;
  const d = `M-32,35 Q-16,${28 - openH * 0.5} 0,35 Q16,${28 - openH * 0.5} 32,35 Q16,${42 + openH} 0,35 Q-16,${42 + openH} -32,35`;
  s.mouthLip.setAttribute('d', d);
  s.mouthInside.setAttribute('d', `M-32,35 Q0,${55 + openH * 2} 32,35`);
  s.mouthInside.setAttribute('opacity', Math.min(1, mouthOpen * 1.5));
}

function drawHandsSVG(ctx, w, h) {
  if (!state.handLandmarks || !state.handLandmarks.multiHandLandmarks) {
    // Fallback sintetico mouse
    if (state.mouseHand.active && state.mouseHand.x != null) {
      drawSyntheticHand(ctx, state.mouseHand.x, state.mouseHand.y);
    }
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  const cw = w / dpr, ch = h / dpr;

  state.handLandmarks.multiHandLandmarks.forEach(lm => {
    // Usiamo landmark 0 (polso) come riferimento della mano
    const x = (1 - lm[0].x) * cw;
    const y = lm[0].y * ch;
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(0.6, 0.6);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.strokeStyle = '#EF4444'; ctx.fillStyle = '#EF4444';
    ctx.beginPath(); ctx.ellipse(0, 0, 22, 28, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    const fingers = [
      { ang: -25, len: 26 },
      { ang: -10, len: 34 },
      { ang: 10, len: 34 },
      { ang: 30, len: 30 },
      { ang: 50, len: 26 },
    ];
    fingers.forEach(f => {
      ctx.save(); ctx.rotate((f.ang * Math.PI) / 180);
      ctx.beginPath(); ctx.moveTo(0, 2); ctx.lineTo(0, -f.len); ctx.stroke();
      ctx.restore();
    });
    ctx.restore();
  });
}

function drawSyntheticHand(ctx, x, y, color = '#EF4444') {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(0.5, 0.5);
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  ctx.strokeStyle = color; ctx.fillStyle = color;
  ctx.beginPath(); ctx.ellipse(0, 0, 22, 28, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  const fingers = [
    { ang: -25, len: 26 },
    { ang: -10, len: 34 },
    { ang: 10, len: 34 },
    { ang: 30, len: 30 },
    { ang: 50, len: 26 },
  ];
  fingers.forEach(f => {
    ctx.save(); ctx.rotate((f.ang * Math.PI) / 180);
    ctx.beginPath(); ctx.moveTo(0, 2); ctx.lineTo(0, -f.len); ctx.stroke();
    ctx.restore();
  });
  ctx.restore();
}

/* ---------- Placeholder sintetico quando non c'è webcam ---------- */
function drawSyntheticPlaceholder(ctx, w, h) {
  ctx.fillStyle = '#0a0a12';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#262C3A';
  ctx.lineWidth = 2;
  ctx.strokeRect(10, 10, w - 20, h - 20);
  ctx.fillStyle = '#94A3B8';
  ctx.font = '14px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Webcam non disponibile', w / 2, h / 2 - 20);
  ctx.fillText('Usa i controlli a destra per disegnare', w / 2, h / 2 + 5);
}

/* ---------- Loop ---------- */
function tick(now) {
  requestAnimationFrame(tick);

  state.frameCount++;
  if (now - state.fpsTime >= 1000) {
    state.fps = state.frameCount;
    state.frameCount = 0;
    state.fpsTime = now;
    els.fpsBadge.textContent = 'FPS: ' + state.fps;
  }

  // Aggiorna lip-sync audio
  const micVol = getMicVolume();
  state.mouthOpenByAudio = Math.min(1, micVol * 2);

  const w = els.canvas.width / (window.devicePixelRatio || 1);
  const h = els.canvas.height / (window.devicePixelRatio || 1);
  const ctx = els.ctx;
  ctx.clearRect(0, 0, w, h);

  // Invia frame a MediaPipe Face Mesh + Hands
  if (state.cameraAvailable && els.video && els.video.readyState >= 2) {
    ctx.drawImage(els.video, 0, 0, w, h);
    els.pipCtx.drawImage(els.video, 0, 0, els.pip.width, els.pip.height);

    if (state.faceMesh && els.video.readyState >= 2) {
      state.faceMesh.send({ image: els.video });
    }
    if (state.hands && els.video.readyState >= 2) {
      state.hands.send({ image: els.video });
    }

    // Applica tracciamento faccia all'SVG
    if (state.faceLandmarks) {
      applyFaceToSVG(state.faceLandmarks);
    }

    // Invia frame all'API Dense-Armor (opzionale)
    if (now - state.lastApiSend > 150) {
      state.lastApiSend = now;
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = 160;
      tempCanvas.height = 120;
      const tctx = tempCanvas.getContext('2d');
      tctx.drawImage(els.video, 0, 0, 160, 120);
      const imgData = tctx.getImageData(0, 0, 160, 120);
      const arr = Array.from(imgData.data);
      const body = JSON.stringify({ array: arr });
      Promise.all([
        fetch('http://127.0.0.1:5000/features/transform', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }).then(r => r.json()).catch(() => ({})),
        fetch('http://127.0.0.1:5000/motion/global', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }).then(r => r.json()).catch(() => ({})),
        fetch('http://127.0.0.1:5000/motion/energy', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }).then(r => r.json()).catch(() => ({})),
      ]).then(([features, motion, energy]) => {
        state.apiParams = { ...features, ...motion, ...energy };
        const motion_max = state.apiParams.motion_max || 0;
        fetch('http://127.0.0.1:5000/adwin/update', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ value: motion_max }) }).then(r => r.json()).catch(() => ({}));
      });
    }

  } else {
    ctx.fillStyle = '#0a0a12';
    ctx.fillRect(0, 0, w, h);
    drawSyntheticPlaceholder(ctx, w, h);

    // Animazione sintetica dell'SVG senza webcam
    const t = now * 0.001;
    const s = state.svg;
    s.head.setAttribute('transform', `translate(320,240) rotate(${Math.sin(t) * 5},0,0)`);
    s.neck.setAttribute('transform', `translate(320,240) rotate(${Math.cos(t) * 3},0,0)`);
    s.mouthLip.setAttribute('d', `M-32,35 Q-16,${28 + Math.sin(t * 2) * 5} 0,35 Q16,${28 + Math.sin(t * 2) * 5} 32,35 Q16,${42 + Math.sin(t * 2) * 5} 0,35 Q-16,${42 + Math.sin(t * 2) * 5} -32,35`);
    s.mouthInside.setAttribute('opacity', Math.abs(Math.sin(t * 2)) * 0.5);
  }

  // Disegna layer utente (se c'è qualcosa disegnato)
  const cur = state.layers[state.currentLayer];
  if (cur && cur.canvas.width > 0) {
    ctx.drawImage(cur.canvas, 0, 0, w, h);
  }

  // Disegna mani
  drawHandsSVG(ctx, w, h);
}

init();

