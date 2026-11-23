/* D-Avatar Studio — Live Input. Webcam + disegno + tracciamento mani (MediaPipe Hands). */

const state = {
  cameraOn: true,
  cleanMode: false,
  video: null,
  stream: null,
  cameraAvailable: false,
  video: null,
  stream: null,
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
  handsResults: null,
  handsReady: false,
  syntheticTime: 0,
  mouseHand: { x: null, y: null, active: false },
};

const els = {};
function $(id) { return document.getElementById(id); }

function init() {
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
  els.crosshair = $('crosshair');
  state.dctx = els.dctx;

  initLayers();
  initEvents();
  resizeCanvases();
  window.addEventListener('resize', resizeCanvases);

  startWebcam().then(() => {
    requestAnimationFrame(tick);
  }).catch(err => {
    console.warn('Webcam non disponibile, uso placeholder sintetico:', err);
    state.cameraAvailable = false;
    els.faceBadge.textContent = 'CAM: NO CAM';
    els.faceBadge.className = 'pill-small pill-red';
    els.handBadge.textContent = 'HAND: --';
    els.handBadge.className = 'pill-small pill-red';
    // Avvia comunque il loop con placeholder sintetico
    state.syntheticTime = 0;
    requestAnimationFrame(tick);
  });

  initHands();
}

function initLayers() {
  state.layers = {};
  const names = ['head', 'eyes-open', 'eyes-closed', 'mouth-open', 'mouth-closed', 'hand-default', 'hand-horns'];
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

async function startWebcam() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    state.video.srcObject = state.stream;
    await new Promise(resolve => { state.video.onloadedmetadata = resolve; });
    state.video.play();
    state.cameraOn = true;
    state.cameraAvailable = true;
    $('webcam-toggle').checked = true;
    els.faceBadge.textContent = 'CAM: OK';
    els.faceBadge.className = 'pill-small pill-green';
  } catch (e) {
    state.cameraOn = false;
    state.cameraAvailable = false;
    throw e;
  }
}

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
    if (!window.Hands) {
      await loadHandsScript();
    }
    const Hands = window.Hands;
    if (!Hands) {
      console.warn('MediaPipe Hands non disponibile');
      state.hands = null;
      state.handsReady = false;
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
  state.handsResults = results;
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    els.handBadge.textContent = 'HAND: OK';
    els.handBadge.className = 'pill-small pill-green';
  } else {
    els.handBadge.textContent = 'HAND: --';
    els.handBadge.className = 'pill-small pill-red';
  }
}

function initEvents() {
  $('webcam-toggle').addEventListener('change', (e) => {
    state.cameraOn = e.target.checked;
    if (!state.cameraOn && state.stream) {
      state.stream.getTracks().forEach(t => t.stop());
      state.stream = null;
    } else if (state.cameraOn && !state.stream) {
      startWebcam();
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

  $('btn-apply').addEventListener('click', () => {
    requestAnimationFrame(tick);
  });

  $('btn-export').addEventListener('click', () => {
    const data = {};
    Object.keys(state.layers).forEach(k => {
      data[k] = state.layers[k].canvas.toDataURL('image/png');
    });
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
      } catch (err) {
        alert('File JSON non valido');
      }
    };
    reader.readAsText(file);
  });

  $('sw-black').addEventListener('click', () => setBg('black'));
  $('sw-green').addEventListener('click', () => setBg('green'));
  $('sw-trans').addEventListener('click', () => setBg('trans'));

  // Retry tracciamento mani (MediaPipe)
  $('btn-retry-hands').addEventListener('click', async () => {
    await retryHands();
  });

  // Tracciamento mano sintetico (fallback quando MediaPipe non è disponibile)
  els.canvas.addEventListener('mousemove', (e) => {
    const rect = els.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    state.mouseHand.x = (e.clientX - rect.left) * dpr;
    state.mouseHand.y = (e.clientY - rect.top) * dpr;
    state.mouseHand.active = true;
  });
  els.canvas.addEventListener('mouseleave', () => {
    state.mouseHand.active = false;
  });
}

function updateCrosshair(x, y, visible) {
  const cr = els.crosshair;
  if (visible) {
    cr.style.left = x + 'px';
    cr.style.top = y + 'px';
    cr.style.display = 'block';
  } else {
    cr.style.display = 'none';
  }
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
  const hasSub = ['eyes', 'mouth', 'hand'].includes(state.currentLayer);
  subs.style.display = hasSub ? 'flex' : 'none';
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

/* ---------- Disegno avatar migliorato ---------- */
function drawAvatar(ctx, w, h, params) {
  const s = w / 400.0;
  const cx = w / 2;
  const cy = h / 2 + (params.breathing || 0);
  const yaw = (params.head_yaw || 0) * 0.02;
  const pitch = (params.head_pitch || 0) * 0.02;
  const roll = (params.head_roll || 0) * 0.02;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(roll);

  // Spalle
  ctx.save();
  ctx.rotate(roll * 0.2);
  ctx.fillStyle = '#151922';
  ctx.beginPath(); ctx.ellipse(0, 60 + pitch * 2, 140, 30, 0, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#111827'; ctx.lineWidth = 3; ctx.stroke();
  ctx.restore();

  // Collo
  ctx.fillStyle = '#FDBA74';
  ctx.beginPath(); ctx.ellipse(0, 20 + pitch * 1.5, 22, 40, 0, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#111827'; ctx.lineWidth = 3; ctx.stroke();

  // Testa
  ctx.save();
  ctx.translate(yaw * 15, pitch * 12);
  ctx.fillStyle = '#FDBA74';
  ctx.beginPath(); ctx.ellipse(0, -10, 100, 115, 0, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#111827'; ctx.lineWidth = 4; ctx.stroke();

  // Capelli
  ctx.fillStyle = '#4A3F35';
  ctx.beginPath(); ctx.ellipse(0, -75, 108, 140, 0, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#111827'; ctx.lineWidth = 3; ctx.stroke();
  ctx.fillStyle = '#8B7355';
  ctx.beginPath(); ctx.ellipse(-35, -65, 18, 12, roll * 0.5, 0, Math.PI); ctx.fill();

  // Occhi
  const eyeY = -45 + pitch * 3;
  const eyeSep = 55 + yaw * 8;
  const eyeW = 26, eyeH = 18;
  const eyeScale = 1 - (params.blink || 0) * 0.85;
  function drawEye(side) {
    ctx.save();
    ctx.translate(side * eyeSep, eyeY + (params.hairBounce || 0) * 0.3);
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath(); ctx.ellipse(0, 0, eyeW, eyeH * eyeScale, 0, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#111827'; ctx.lineWidth = 3; ctx.stroke();
    ctx.fillStyle = '#06B6D4';
    ctx.beginPath(); ctx.arc((params.gaze_x || 0) * 10, (params.gaze_y || 0) * 6, 8 * eyeScale, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#000000';
    ctx.beginPath(); ctx.arc((params.gaze_x || 0) * 10 + 1, (params.gaze_y || 0) * 6 + 1, 4 * eyeScale, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath(); ctx.arc((params.gaze_x || 0) * 10 - 2.5, (params.gaze_y || 0) * 6 - 2, 2, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#111827'; ctx.lineWidth = 3; ctx.beginPath();
    ctx.moveTo(-eyeW, -eyeH * eyeScale + 2);
    ctx.bezierCurveTo(-eyeW * 0.6, -eyeH * eyeScale - 6, eyeW * 0.6, -eyeH * eyeScale - 6, eyeW, -eyeH * eyeScale + 2);
    ctx.stroke();
    ctx.restore();
  }
  drawEye(-1); drawEye(1);

  // Sopracciglia
  ctx.save();
  ctx.translate(yaw * 10, pitch * 5);
  [-1, 1].forEach(side => {
    ctx.save();
    ctx.translate(side * eyeSep, -70 + (params.hairBounce || 0) * 0.4);
    ctx.rotate((params.smile || 0) * 0.15 * side + roll * 0.3);
    ctx.strokeStyle = '#4A3F35'; ctx.lineWidth = 5; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(-20, -5); ctx.lineTo(20, -5); ctx.stroke();
    ctx.restore();
  });
  ctx.restore();

  // Naso
  ctx.save();
  ctx.translate(0, -15 + pitch * 2);
  ctx.strokeStyle = '#111827'; ctx.lineWidth = 3; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(-4, 6); ctx.lineTo(0, -8); ctx.lineTo(4, 6); ctx.stroke();
  ctx.fillStyle = '#FDBA74'; ctx.beginPath(); ctx.arc(0, 6, 5, 0, Math.PI * 2); ctx.fill();
  ctx.restore();

  // Bocca
  const mouthY = 30 + pitch * 4;
  const mouthW = 30;
  ctx.save();
  ctx.translate(0, mouthY + (params.hairBounce || 0) * 0.5);
  ctx.fillStyle = '#111827';
  ctx.beginPath(); ctx.ellipse(0, 0, mouthW, 14, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#F472B6';
  ctx.beginPath(); ctx.moveTo(-mouthW, 0);
  ctx.bezierCurveTo(-mouthW * 0.6, -8, mouthW * 0.6, -8, mouthW, 0);
  ctx.bezierCurveTo(mouthW * 0.6, 4, -mouthW * 0.6, 4, -mouthW, 0);
  ctx.fill();
  if ((params.mouth_open || 0) > 0.2) {
    const oh = 16 * params.mouth_open;
    ctx.fillStyle = '#7F1D1D';
    ctx.beginPath(); ctx.ellipse(0, oh / 2 + 2, mouthW * 0.85, oh, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#FFFFFF'; ctx.fillRect(-10, oh / 2 - 6, 6, 6); ctx.fillRect(4, oh / 2 - 6, 6, 6);
    ctx.fillStyle = '#F472B6'; ctx.beginPath(); ctx.ellipse(0, oh / 2 + 4, mouthW * 0.6, 8, 0, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();

  ctx.restore(); ctx.restore();
  ctx.strokeStyle = '#111827'; ctx.lineWidth = 4; ctx.beginPath(); ctx.ellipse(0, -10, 102, 117, roll, 0, Math.PI * 2); ctx.stroke();

  // Mani da tracciamento (MediaPipe o sintetico)
  drawHands(ctx, w, h);
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

function drawHands(ctx, w, h) {
  // Mani da MediaPipe (se disponibile)
  if (state.handsResults && state.handsResults.multiHandLandmarks) {
    const dpr = window.devicePixelRatio || 1;
    const cw = w / dpr, ch = h / dpr;
    state.handsResults.multiHandLandmarks.forEach(lm => {
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

  // Mano sintetica dal mouse (fallback quando MediaPipe non c'è)
  else if (state.mouseHand.active && state.mouseHand.x != null) {
    drawSyntheticHand(ctx, state.mouseHand.x, state.mouseHand.y);
  }
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

  const w = els.canvas.width / (window.devicePixelRatio || 1);
  const h = els.canvas.height / (window.devicePixelRatio || 1);
  const ctx = els.ctx;
  ctx.clearRect(0, 0, w, h);

  if (state.cameraAvailable && state.video && state.video.readyState >= 2) {
    ctx.drawImage(state.video, 0, 0, w, h);
    els.pipCtx.drawImage(state.video, 0, 0, els.pip.width, els.pip.height);
    if (state.hands && state.video.readyState >= 2) {
      state.hands.send({ image: state.video });
    }
  } else {
    ctx.fillStyle = '#0a0a12';
    ctx.fillRect(0, 0, w, h);
    drawSyntheticPlaceholder(ctx, w, h);
  }

  // Disegna layer utente (se c'è qualcosa disegnato)
  const cur = state.layers[state.currentLayer];
  if (cur && cur.canvas.width > 0) {
    ctx.drawImage(cur.canvas, 0, 0, w, h);
  }

  // Disegna mani SOLO se ci sono landmark reali (MediaPipe o mouse)
  drawHands(ctx, w, h);
}

init();
