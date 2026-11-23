/* D-Avatar Studio — Live Input. Nessuna dipendenza esterna. Webcam + disegno. */

const state = {
  cameraOn: true,
  cleanMode: false,
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
  fpsTime: 0
};

const els = {};
function $ (id) { return document.getElementById(id); }

function init() {
  // Riferimenti DOM
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
    console.error('Webcam non avviata:', err);
    els.faceBadge.textContent = 'CAM: NO CAM';
    els.faceBadge.className = 'pill-small pill-red';
    // Continua comunque così si possono testare i pulsanti
    requestAnimationFrame(tick);
  });
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
  // Disegna un volto di default se vuoto (così si vede qualcosa)
  drawDefaultFace();
}

function drawDefaultFace() {
  const l = state.layers;
  // Testa
  const c = l.head.ctx;
  c.fillStyle = '#FDBA74';
  c.beginPath(); c.arc(150, 150, 110, 0, Math.PI * 2); c.fill();
  c.strokeStyle = '#111827'; c.lineWidth = 4; c.stroke();
  // Occhi
  c.fillStyle = '#FFFFFF';
  c.beginPath(); c.arc(115, 130, 22, 0, Math.PI * 2); c.fill();
  c.beginPath(); c.arc(185, 130, 22, 0, Math.PI * 2); c.fill();
  c.fillStyle = '#06B6D4';
  c.beginPath(); c.arc(115, 130, 10, 0, Math.PI * 2); c.fill();
  c.beginPath(); c.arc(185, 130, 10, 0, Math.PI * 2); c.fill();
  // Bocca
  c.fillStyle = '#F472B6';
  c.beginPath(); c.arc(150, 180, 25, 0, Math.PI); c.fill();

  // Occhi chiusi
  const ec = l['eyes-closed'].ctx;
  ec.fillStyle = '#FFFFFF';
  ec.beginPath(); ec.arc(115, 130, 22, 0, Math.PI * 2); ec.fill();
  ec.beginPath(); ec.arc(185, 130, 22, 0, Math.PI * 2); ec.fill();
  ec.strokeStyle = '#111827'; ec.lineWidth = 4;
  ec.beginPath(); ec.moveTo(93, 130); ec.lineTo(137, 130); ec.stroke();
  ec.beginPath(); ec.moveTo(163, 130); ec.lineTo(207, 130); ec.stroke();

  // Bocca aperta
  const mo = l['mouth-open'].ctx;
  mo.fillStyle = '#7F1D1D';
  mo.beginPath(); mo.arc(150, 185, 28, 0, Math.PI * 2); mo.fill();
  mo.fillStyle = '#FFFFFF';
  mo.fillRect(138, 175, 10, 12);
  mo.fillRect(152, 175, 10, 12);

  // Bocca chiusa
  const mc = l['mouth-closed'].ctx;
  mc.fillStyle = '#F472B6';
  mc.beginPath(); mc.arc(150, 180, 25, 0, Math.PI); mc.fill();

  // Mano default
  const hd = l['hand-default'].ctx;
  hd.fillStyle = '#FDBA74';
  hd.beginPath(); hd.arc(150, 150, 24, 0, Math.PI * 2); hd.fill();
  hd.strokeStyle = '#111827'; hd.lineWidth = 3; hd.stroke();

  // Mano corna
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

  // Salva lo stato del pennello prima che cambiare le dimensioni del canvas
  // lo resetti (cambiare width/height azzera il contesto 2D).
  const saved = {
    strokeStyle: state.dctx?.strokeStyle ?? '#ffffff',
    lineWidth: state.dctx?.lineWidth ?? 8,
    lineCap: state.dctx?.lineCap ?? 'round',
    lineJoin: state.dctx?.lineJoin ?? 'round',
  };

  els.draw.width = 300;
  els.draw.height = 300;
  els.dctx.setTransform(1, 0, 0, 1, 0, 0);

  // Ripristina lo stato del pennello dopo il resize.
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
    $('webcam-toggle').checked = true;
    els.faceBadge.textContent = 'CAM: OK';
    els.faceBadge.className = 'pill-small pill-green';
  } catch (e) {
    state.cameraOn = false;
    throw e;
  }
}

function initEvents() {
  // Webcam toggle
  $('webcam-toggle').addEventListener('change', (e) => {
    state.cameraOn = e.target.checked;
    if (!state.cameraOn && state.stream) {
      state.stream.getTracks().forEach(t => t.stop());
      state.stream = null;
    } else if (state.cameraOn && !state.stream) {
      startWebcam();
    }
  });

  // Clean mode
  $('btn-clean').addEventListener('click', () => {
    state.cleanMode = !state.cleanMode;
    applyCleanMode();
  });

  // Disegno canvas
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

  // Pennello
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

  $('btn-undo').addEventListener('click', () => {
    clearLayer(state.currentLayer);
  });
  $('btn-clear').addEventListener('click', () => {
    clearLayer(state.currentLayer);
  });

  $('btn-load-png').addEventListener('click', () => {
    $('png-file').click();
  });
  $('png-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
      state.dctx.drawImage(img, 0, 0, els.draw.width, els.draw.height);
    };
    img.src = URL.createObjectURL(file);
  });

  // Tabs layer
  document.querySelectorAll('#layer-tabs .tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('#layer-tabs .tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.currentLayer = tab.dataset.layer;
      updateSubtabs();
    });
  });

  // Subtabs
  document.querySelectorAll('#subtabs .tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('#subtabs .tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
    });
  });

  // Salva PNG
  $('save-png').addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = 'avatar-' + state.currentLayer + '-' + Date.now() + '.png';
    link.href = els.draw.toDataURL('image/png');
    link.click();
  });

  // Applica
  $('btn-apply').addEventListener('click', () => {
    // Il disegno viene già renderato nel tick, ma forziamo un redraw
    requestAnimationFrame(tick);
  });

  // Esporta / Importa JSON
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

  $('btn-import').addEventListener('click', () => {
    $('json-import').click();
  });
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

  // Sfondo
  $('sw-black').addEventListener('click', () => setBg('black'));
  $('sw-green').addEventListener('click', () => setBg('green'));
  $('sw-trans').addEventListener('click', () => setBg('trans'));
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
    // Se è il layer head, ridisegna default
    if (name === 'head') drawDefaultFace();
  }
}

function updateSubtabs() {
  const subs = $('subtabs');
  // Mostra subtabs solo per layer che li hanno
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

function tick(now) {
  requestAnimationFrame(tick);

  // FPS
  state.frameCount++;
  if (now - state.fpsTime >= 1000) {
    state.fps = state.frameCount;
    state.frameCount = 0;
    state.fpsTime = now;
    els.fpsBadge.textContent = 'FPS: ' + state.fps;
  }

  // Disegna webcam sul canvas principale (se attiva)
  if (state.cameraOn && state.video && state.video.readyState >= 2) {
    els.ctx.drawImage(state.video, 0, 0, els.canvas.width / (window.devicePixelRatio || 1), els.canvas.height / (window.devicePixelRatio || 1));
    // PiP
    els.pipCtx.drawImage(state.video, 0, 0, els.pip.width, els.pip.height);
    els.handBadge.textContent = 'HAND: OK';
    els.handBadge.className = 'pill-small pill-green';
  } else {
    els.ctx.fillStyle = '#0a0a12';
    els.ctx.fillRect(0, 0, els.canvas.width / (window.devicePixelRatio || 1), els.canvas.height / (window.devicePixelRatio || 1));
    els.handBadge.textContent = 'HAND: --';
    els.handBadge.className = 'pill-small pill-red';
  }

  // Render layer disegno sopra
  const cur = state.layers[state.currentLayer];
  if (cur && cur.canvas.width > 0) {
    els.ctx.drawImage(cur.canvas, 0, 0, els.canvas.width / (window.devicePixelRatio || 1), els.canvas.height / (window.devicePixelRatio || 1));
  }

  // Overlay occhi/bocca basato su landmark (se mai attivi)
  if (state.faceLandmarks) {
    // placeholder per futuro tracking
  }
  if (state.handLandmarks) {
    // placeholder
  }
}

init();
