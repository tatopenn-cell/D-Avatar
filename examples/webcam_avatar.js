/* D-Avatar Studio — Live Input. Webcam + Live2D + Face Mesh + Lip-sync + disegno + mani. */

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
  // Effetti TikTok
  currentEffect: 'none',
  // Live2D
  live2dApp: null,
  live2dModel: null,
  live2dReady: false,
  // Parametri API Dense-Armor (opzionali)
  apiParams: {},
  lastApiSend: 0,
};

const els = {};
function $(id) { return document.getElementById(id); }

function init() {
  const required = ['main-canvas','draw-canvas','webcam-video','pip-cam','badge-face','badge-hand','badge-fps','badge-mic','badge-live2d','crosshair','live2d-canvas'];
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
  els.live2dCanvas = $('live2d-canvas');
  els.fxCanvas = $('fx-canvas');
  els.fxCtx = els.fxCanvas.getContext('2d');
  els.faceBadge = $('badge-face');
  els.handBadge = $('badge-hand');
  els.fpsBadge = $('badge-fps');
  els.micBadge = $('badge-mic');
  els.live2dBadge = $('badge-live2d');
  els.crosshair = $('crosshair');
  state.dctx = els.dctx;

  initLayers();
  initEvents();
  resizeCanvases();
  window.addEventListener('resize', resizeCanvases);

  initMic();
  initLive2D();

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

  // Ridimensiona Live2D canvas
  if (els.live2dCanvas) {
    els.live2dCanvas.width = w * dpr;
    els.live2dCanvas.height = h * dpr;
    els.live2dCanvas.style.width = w + 'px';
    els.live2dCanvas.style.height = h + 'px';
  }
  // Ridimensiona canvas effetti
  if (els.fxCanvas) {
    els.fxCanvas.width = w * dpr;
    els.fxCanvas.height = h * dpr;
    els.fxCanvas.style.width = w + 'px';
    els.fxCanvas.style.height = h + 'px';
  }
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
  return Math.min(1, avg / 120);
}

/* ---------- Live2D (pixi-live2d-display) ---------- */
async function initLive2D() {
  try {
    if (!window.PIXI || !window.PIXI.Live2D) {
      throw new Error('PixiJS / Live2D non caricati');
    }
    state.live2dApp = new PIXI.Application({
      view: els.live2dCanvas,
      resizeTo: window,
      autoStart: true,
      transparent: true,
      backgroundAlpha: 0,
    });
    state.live2dApp.stage.removeChildren();

    const model = await PIXI.Live2D.Model.from('modelli/hiyori/hiyori.model3.json');
    state.live2dModel = model;
    state.live2dApp.stage.addChild(model);

    // Centra e scala il modello nello stage
    const scale = 0.6;
    model.scale.set(scale, scale);
    model.position.set(
      state.live2dApp.screen.width / 2,
      state.live2dApp.screen.height / 2 + 40
    );

    state.live2dReady = true;
    els.live2dBadge.textContent = 'LIVE2D: OK';
    els.live2dBadge.className = 'pill-small pill-green';
  } catch (e) {
    console.warn('Impossibile caricare Live2D:', e);
    state.live2dReady = false;
    els.live2dBadge.textContent = 'LIVE2D: ERR';
    els.live2dBadge.className = 'pill-small pill-red';
  }
}

function setLive2DParam(paramId, value) {
  if (!state.live2dReady || !state.live2dModel) return;
  try {
    const core = state.live2dModel.coreModel;
    if (core && typeof core.setParameterValue === 'function') {
      core.setParameterValue(paramId, value);
      return;
    }
    if (typeof state.live2dModel.setParameterValue === 'function') {
      state.live2dModel.setParameterValue(paramId, value);
    }
  } catch (e) {
    // Ignora errori su parametri non esistenti nel modello
  }
}

function applyFaceToLive2D(lm) {
  if (!lm || !lm.multiFaceLandmarks || lm.multiFaceLandmarks.length === 0) return;
  const face = lm.multiFaceLandmarks[0];

  // Landmark chiave Face Mesh
  const nose = face[1];
  const chin = face[152];
  const forehead = face[10];
  const lipUp = face[13];
  const lipDown = face[14];
  const eyeL = face[33];
  const eyeR = face[263];
  const browL = face[105];
  const browR = face[334];
  const eyeTopL = face[159];
  const eyeBotL = face[145];
  const eyeTopR = face[386];
  const eyeBotR = face[374];

  // --- Testa: rotazione X/Y/Z ---
  const faceCx = (eyeL.x + eyeR.x) / 2;
  const faceCy = (eyeL.y + eyeR.y + nose.y + chin.y) / 4;
  const yaw = (nose.x - faceCx) * 60;   // gradi ~±60
  const pitch = (nose.y - faceCy) * 50; // gradi ~±50
  const roll = 0;

  setLive2DParam('ParamAngleX', yaw * 0.8);
  setLive2DParam('ParamAngleY', pitch * 0.8);
  setLive2DParam('ParamAngleZ', roll * 0.5);

  // --- Occhi: apertura ---
  const openL = Math.max(0, (eyeBotL.y - eyeTopL.y) / 0.015);
  const openR = Math.max(0, (eyeBotR.y - eyeTopR.y) / 0.015);
  const eyeOpenL = Math.min(1, openL * 2.5);
  const eyeOpenR = Math.min(1, openR * 2.5);

  setLive2DParam('ParamEyeLOpen', eyeOpenL);
  setLive2DParam('ParamEyeROpen', eyeOpenR);

  // --- Sopracciglia: alzate/abbassate ---
  const browLift = (browL.y + browR.y) / 2 - (forehead.y + 0.02);
  const browParam = Math.max(-1, Math.min(1, browLift * 40));
  setLive2DParam('ParamBrowLY', browParam);
  setLive2DParam('ParamBrowRY', browParam);

  // --- Bocca: apertura (faccia + audio lip-sync) ---
  const mouthDist = Math.max(0, lipDown.y - lipUp.y);
  const mouthFromFace = Math.min(1, mouthDist / 0.04);
  const mouthFromAudio = state.mouthOpenByAudio;
  const mouthOpen = Math.max(mouthFromFace, mouthFromAudio);
  setLive2DParam('ParamMouthOpenY', mouthOpen);
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
    // Se il CDN è già stato caricato in precedenza, window.Hands esiste già
    if (!window.Hands) {
      await loadHandsScript();
    }
    const Hands = window.Hands;
    if (!Hands) {
      // CDN bloccato: non è un errore fatale, le mani restano disattivate
      state.hands = null;
      state.handsReady = false;
      els.handBadge.textContent = 'HAND: N/D';
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
    els.handBadge.textContent = 'HAND: N/D';
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
      els.faceBadge.textContent = 'FACE: N/D';
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
    els.faceBadge.textContent = 'FACE: N/D';
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

/* ---------- Disegno mani sintetico (fallback mouse) ---------- */
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

/* ---------- Effetti TikTok sul volto (basati su Face Mesh landmark) ---------- */
function getFaceBox(lm, w, h) {
  if (!lm || !lm.multiFaceLandmarks || lm.multiFaceLandmarks.length === 0) return null;
  const face = lm.multiFaceLandmarks[0];
  // Landmark del contorno viso (oval): 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
  const idx = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
  let minX=1,minY=1,maxX=0,maxY=0;
  idx.forEach(i=>{
    const p=face[i];
    if(p.x<minX)minX=p.x; if(p.x>maxX)maxX=p.x;
    if(p.y<minY)minY=p.y; if(p.y>maxY)maxY=p.y;
  });
  const pad=0.15;
  const bw=(maxX-minX), bh=(maxY-minY);
  return {
    x:Math.max(0,minX-bpad)*w,
    y:Math.max(0,minY-bpad)*h,
    w:Math.min(1,maxX+bpad)*w - Math.max(0,minX-bpad)*w,
    h:Math.min(1,maxY+bpad)*h - Math.max(0,minY-bpad)*h,
    cx:(minX+maxX)/2*w,
    cy:(minY+maxY)/2*h,
  };
}
const bpad=0.15;

function drawEyeOverlay(ctx, cx, cy, r, color='#00FF00') {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.strokeStyle=color; ctx.lineWidth=4;
  ctx.beginPath(); ctx.arc(0,0,r,0,Math.PI*2); ctx.stroke();
  ctx.beginPath(); ctx.arc(0,0,r*0.6,0,Math.PI*2); ctx.stroke();
  ctx.restore();
}

function applyEffect(ctx, w, h) {
  const eff = state.currentEffect;
  if (eff === 'none') return;

  const lm = state.faceLandmarks;
  const box = lm ? getFaceBox(lm, w, h) : null;

  ctx.save();

  // Effetti filtro: prima disegno il video sul canvas effetti
  const filterEffects = ['mirror','pixelate','sepia','invert','blur','glitch'];
  if (filterEffects.includes(eff) && box) {
    ctx.drawImage(els.video, box.x, box.y, box.w, box.h, box.x, box.y, box.w, box.h);
  }

  if (eff === 'mirror') {
    if (box) {
      ctx.translate(box.x + box.w, box.y);
      ctx.scale(-1, 1);
      ctx.drawImage(els.video, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);
    }
    ctx.restore();
    return;
  }

  if (eff === 'pixelate' && box) {
    const px = 14;
    const cols = Math.ceil(box.w/px), rows = Math.ceil(box.h/px);
    for (let ry=0; ry<rows; ry++) {
      for (let rx=0; rx<cols; rx++) {
        const sx = box.x + rx*px;
        const sy = box.y + ry*px;
        const pw = Math.min(px, box.w - rx*px);
        const ph = Math.min(px, box.h - ry*px);
        const p = ctx.getImageData(sx, sy, 1, 1).data;
        ctx.fillStyle = `rgb(${p[0]},${p[1]},${p[2]})`;
        ctx.fillRect(sx, sy, pw, ph);
      }
    }
    ctx.restore();
    return;
  }

  if (eff === 'sepia' && box) {
    const imgData = ctx.getImageData(box.x, box.y, box.w, box.h);
    const d = imgData.data;
    for (let i=0; i<d.length; i+=4) {
      const r=d[i], g=d[i+1], b=d[i+2];
      d[i]=Math.min(255,(r*0.393)+(g*0.769)+(b*0.189));
      d[i+1]=Math.min(255,(r*0.349)+(g*0.686)+(b*0.168));
      d[i+2]=Math.min(255,(r*0.272)+(g*0.534)+(b*0.131));
    }
    ctx.putImageData(imgData, box.x, box.y);
    ctx.restore();
    return;
  }

  if (eff === 'invert' && box) {
    const imgData = ctx.getImageData(box.x, box.y, box.w, box.h);
    const d = imgData.data;
    for (let i=0; i<d.length; i+=4) {
      d[i]=255-d[i]; d[i+1]=255-d[i+1]; d[i+2]=255-d[i+2];
    }
    ctx.putImageData(imgData, box.x, box.y);
    ctx.restore();
    return;
  }

  if (eff === 'blur' && box) {
    ctx.filter='blur(10px)';
    ctx.drawImage(els.video, box.x, box.y, box.w, box.h, box.x, box.y, box.w, box.h);
    ctx.filter='none';
    ctx.restore();
    return;
  }

  if (eff === 'glitch' && box) {
    const t=Date.now()*0.002;
    const offset=Math.sin(t*6)*12;
    ctx.drawImage(els.video, box.x+offset, box.y, box.w, box.h, box.x, box.y, box.w, box.h);
    ctx.drawImage(els.video, box.x-offset, box.y+15, box.w, box.h, box.x, box.y+15, box.w, box.h);
    ctx.fillStyle='rgba(255,0,0,0.15)';
    ctx.fillRect(box.x, box.y, box.w, box.h);
    ctx.restore();
    return;
  }

  // Effetti overlay (disegnati sopra il video già presente)
  if (!box) { ctx.restore(); return; }

  const {cx, cy} = box;
  const faceW = box.w, faceH = box.h;

  if (eff === 'cry') {
    ctx.fillStyle='#06B6D4';
    const t=Date.now();
    const eyeL = lm.multiFaceLandmarks[0][33];
    const eyeR = lm.multiFaceLandmarks[0][263];
    [eyeL, eyeR].forEach((e, idx) => {
      const ex = e.x*w, ey = e.y*h;
      for (let i=0; i<3; i++) {
        const tx = ex + (idx===0?-8:8) + Math.sin(t*0.005+idx+i)*6;
        const ty = ey + 10 + (i*14) + Math.sin(t*0.003+idx+i)*4;
        ctx.beginPath(); ctx.arc(tx, ty, 5, 0, Math.PI*2); ctx.fill();
        ctx.beginPath(); ctx.moveTo(tx, ty+5); ctx.lineTo(tx, ty+11);
        ctx.strokeStyle='#06B6D4'; ctx.lineWidth=2; ctx.stroke();
      }
    });
    ctx.restore();
    return;
  }

  if (eff === 'bigeyes') {
    const r = faceW * 0.18;
    const eyeL = lm.multiFaceLandmarks[0][33];
    const eyeR = lm.multiFaceLandmarks[0][263];
    drawEyeOverlay(ctx, eyeL.x*w, eyeL.y*h, r);
    drawEyeOverlay(ctx, eyeR.x*w, eyeR.y*h, r);
    ctx.restore();
    return;
  }

  if (eff === 'bighead') {
    ctx.strokeStyle='#FF00FF'; ctx.lineWidth=6;
    ctx.beginPath(); ctx.arc(cx, cy - faceH*0.1, faceW*0.65, 0, Math.PI*2); ctx.stroke();
    ctx.restore();
    return;
  }

  ctx.restore();
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

  // Selezione effetti
  document.querySelectorAll('.eff-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.eff-btn').forEach(b => b.style.background = 'transparent');
      btn.style.background = 'var(--accent-primary)';
      state.currentEffect = btn.dataset.effect;
    });
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

  // Disegna webcam sul canvas principale
  if (state.cameraAvailable && els.video && els.video.readyState >= 2) {
    ctx.drawImage(els.video, 0, 0, w, h);
    els.pipCtx.drawImage(els.video, 0, 0, els.pip.width, els.pip.height);

    // Effetti sul canvas sopra il Live2D
    els.fxCtx.clearRect(0, 0, w, h);
    applyEffect(els.fxCtx, w, h);

    // Invia frame a MediaPipe Face Mesh + Hands
    if (state.faceMesh && els.video.readyState >= 2) {
      state.faceMesh.send({ image: els.video });
    }
    if (state.hands && els.video.readyState >= 2) {
      state.hands.send({ image: els.video });
    }

    // Applica tracciamento faccia a Live2D
    if (state.faceLandmarks) {
      applyFaceToLive2D(state.faceLandmarks);
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

    // Animazione sintetica Live2D senza webcam
    if (state.live2dReady) {
      const t = now * 0.001;
      setLive2DParam('ParamAngleX', Math.sin(t) * 10);
      setLive2DParam('ParamAngleY', Math.cos(t) * 5);
      setLive2DParam('ParamEyeLOpen', 1);
      setLive2DParam('ParamEyeROpen', 1);
      setLive2DParam('ParamMouthOpenY', Math.abs(Math.sin(t * 2)) * 0.5);
    }
  }

  // Disegna layer utente (se c'è qualcosa disegnato)
  const cur = state.layers[state.currentLayer];
  if (cur && cur.canvas.width > 0) {
    ctx.drawImage(cur.canvas, 0, 0, w, h);
  }

  // Disegna mani (solo fallback sintetico, Live2D gestisce le proprie se previsto)
  if (state.mouseHand.active && state.mouseHand.x != null) {
    drawSyntheticHand(ctx, state.mouseHand.x, state.mouseHand.y);
  }
}

init();

