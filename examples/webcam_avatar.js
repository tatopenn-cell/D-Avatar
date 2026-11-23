/* D-Avatar Studio — Live Input con MediaPipe FaceLandmarker + HandLandmarker */

let landmarkerFace = null;
let landmarkerHand = null;
let video = null;
let isCameraOn = true;
let isCleanMode = false;
let isDrawing = false;
let currentLayer = 'head';
let layers = {};
let brushSize = 8;
let brushColor = '#ffffff';
let faceLandmarks = null;
let handLandmarks = null;
let mainCtx = null;
let drawCtx = null;
let mainCanvas = null;
let drawCanvas = null;

async function init() {
  mainCanvas = document.getElementById('main-canvas');
  drawCanvas = document.getElementById('draw-canvas');
  mainCtx = mainCanvas.getContext('2d', { willReadFrequently: true });
  drawCtx = drawCanvas.getContext('2d', { willReadFrequently: true });

  initLayers();
  initEvents();
  await initLandmarkers();
  await startWebcam();
  requestAnimationFrame(tick);
}

function initLayers() {
  layers = {};
  ['head','eyes-open','eyes-closed','mouth-open','mouth-closed','hand-default','hand-horns'].forEach(k => {
    const c = document.createElement('canvas');
    c.width = drawCanvas.width;
    c.height = drawCanvas.height;
    layers[k] = { canvas: c, ctx: c.getContext('2d', { willReadFrequently: true }) };
  });
}

function initEvents() {
  document.getElementById('webcam-toggle').addEventListener('change', (e) => {
    isCameraOn = e.target.checked;
  });

  document.getElementById('btn-clean').addEventListener('click', () => {
    isCleanMode = !isCleanMode;
    if (isCleanMode) {
      document.querySelector('header').style.display = 'none';
      document.querySelector('.panel-right').style.display = 'none';
      document.querySelector('.pad-wrap').style.display = 'none';
      document.querySelector('#canvas-wrap').style.border = 'none';
      document.querySelector('.hud').style.display = 'none';
    } else {
      document.querySelector('header').style.display = '';
      document.querySelector('.panel-right').style.display = '';
      document.querySelector('.pad-wrap').style.display = '';
      document.querySelector('#canvas-wrap').style.border = '1px solid var(--border-subtle)';
      document.querySelector('.hud').style.display = '';
    }
  });

  document.getElementById('btn-undo').addEventListener('click', () => {
    layers[currentLayer].ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  });

  document.getElementById('btn-clear').addEventListener('click', () => {
    layers[currentLayer].ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  });

  document.getElementById('btn-save-png').addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = 'avatar-' + currentLayer + '-' + Date.now() + '.png';
    link.href = drawCanvas.toDataURL('image/png');
    link.click();
  });

  document.getElementById('btn-load-png').addEventListener('click', () => {
    document.getElementById('png-file').click();
  });
  document.getElementById('png-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
      layers[currentLayer].ctx.drawImage(img, 0, 0, drawCanvas.width, drawCanvas.height);
    };
    img.src = URL.createObjectURL(file);
  });

  document.getElementById('btn-apply').addEventListener('click', () => {
    // apply simply refreshes the canvas rendering
  });

  document.getElementById('btn-export').addEventListener('click', () => {
    const data = {};
    Object.keys(layers).forEach(k => {
      data[k] = layers[k].canvas.toDataURL('image/png');
    });
    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'd-avatar-character.json';
    a.click();
    URL.revokeObjectURL(url);
  });

  document.getElementById('json-import').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const data = JSON.parse(ev.target.result);
      Object.keys(data).forEach(k => {
        if (layers[k] && data[k]) {
          const img = new Image();
          img.onload = () => layers[k].ctx.drawImage(img, 0, 0, drawCanvas.width, drawCanvas.height);
          img.src = data[k];
        }
      });
    };
    reader.readAsText(file);
  });

  document.getElementById('thresh-eye').addEventListener('input', (e) => {
    document.getElementById('val-eye').textContent = e.target.value;
  });
  document.getElementById('thresh-mouth').addEventListener('input', (e) => {
    document.getElementById('val-mouth').textContent = e.target.value;
  });
  document.getElementById('thresh-head').addEventListener('input', (e) => {
    document.getElementById('val-head').textContent = e.target.value;
  });

  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentLayer = tab.dataset.layer;
    });
  });

  document.querySelectorAll('.c-swatch').forEach(sw => {
    sw.addEventListener('click', () => {
      document.querySelectorAll('.c-swatch').forEach(s => s.classList.remove('active'));
      sw.classList.add('active');
      brushColor = sw.dataset.color;
    });
  });
  document.getElementById('color-picker').addEventListener('input', (e) => {
    brushColor = e.target.value;
    document.querySelectorAll('.c-swatch').forEach(s => s.classList.remove('active'));
  });

  drawCanvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const rect = drawCanvas.getBoundingClientRect();
    drawCtx.beginPath();
    drawCtx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    crosshair.style.left = e.clientX + 'px';
    crosshair.style.top = e.clientY + 'px';
    crosshair.style.display = 'block';
  });
  drawCanvas.addEventListener('mousemove', (e) => {
    crosshair.style.left = e.clientX + 'px';
    crosshair.style.top = e.clientY + 'px';
    if (!isDrawing) return;
    const rect = drawCanvas.getBoundingClientRect();
    drawCtx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    drawCtx.stroke();
  });
  drawCanvas.addEventListener('mouseup', () => {
    isDrawing = false;
    crosshair.style.display = 'none';
  });
  drawCanvas.addEventListener('mouseleave', () => {
    isDrawing = false;
    crosshair.style.display = 'none';
  });

  drawCtx.strokeStyle = brushColor;
  drawCtx.lineWidth = brushSize;
  drawCtx.lineCap = 'round';
  drawCtx.lineJoin = 'round';

  document.getElementById('color-picker').addEventListener('input', (e) => {
    brushColor = e.target.value;
    drawCtx.strokeStyle = brushColor;
    document.querySelectorAll('.c-swatch').forEach(s => s.classList.remove('active'));
  });
  document.getElementById('brush-size').addEventListener('input', (e) => {
    brushSize = parseInt(e.target.value);
    drawCtx.lineWidth = brushSize;
  });
}

async function startWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  video = document.createElement('video');
  video.srcObject = stream;
  video.playbackRate = 1;
  video.play();
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      requestAnimationFrame(tick);
      resolve();
    };
  });
}

async function initLandmarkers() {
  const FaceLandmarker = window.FaceLandmarker;
  const HandLandmarker = window.HandLandmarker;

  const faceOptions = {
    baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' },
    runningMode: 'VIDEO',
    numFaces: 1,
  };
  landmarkerFace = await FaceLandmarker.createFromOptions(faceOptions);

  const handOptions = {
    baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/hand_landmarker/float16/1/hand_landmarker.task' },
    runningMode: 'VIDEO',
    numHands: 1,
  };
  landmarkerHand = await HandLandmarker.createFromOptions(handOptions);
}

function tick() {
  requestAnimationFrame(tick);
  if (!isCameraOn || !video || video.readyState < 2) return;

  mainCtx.drawImage(video, 0, 0, mainCanvas.width, mainCanvas.height);

  const now = performance.now();
  if (!landmarkerFace._lastTime || now - landmarkerFace._lastTime > 100) {
    landmarkerFace.detectForVideo(video, now).then((result) => {
      faceLandmarks = result.faceLandmarks[0] || null;
    });
    landmarkerHand.detectForVideo(video, now).then((result) => {
      handLandmarks = result.landmarks[0] || null;
    });
    landmarkerFace._lastTime = now;
  }

  drawWebcamPip(video);
  renderAvatar();
}

function drawWebcamPip(video) {
  const pip = document.getElementById('pip-cam');
  pip.getContext('2d').drawImage(video, 0, 0, pip.width, pip.height);
}

function renderAvatar() {
  mainCtx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);

  if (layers['head'] && layers['head'].canvas.width > 0) {
    mainCtx.drawImage(layers['head'].canvas, 0, 0, mainCanvas.width, mainCanvas.height);
  } else {
    mainCtx.fillStyle = '#f0c0a0';
    mainCtx.beginPath();
    mainCtx.arc(mainCanvas.width/2, mainCanvas.height/2, 150, 0, Math.PI*2);
    mainCtx.fill();
  }

  if (faceLandmarks) {
    const leftOpen = isEyeOpen(468, 386);
    const rightOpen = isEyeOpen(474, 474);
    const mouthOpen = isMouthOpen();

    if (layers['eyes-open'] && layers['eyes-closed']) {
      const src = (leftOpen && rightOpen) ? layers['eyes-open'] : layers['eyes-closed'];
      if (src.canvas.width > 0) mainCtx.drawImage(src.canvas, 0, 0, mainCanvas.width, mainCanvas.height);
    }

    if (layers['mouth-open'] && layers['mouth-closed']) {
      const src = mouthOpen ? layers['mouth-open'] : layers['mouth-closed'];
      if (src.canvas.width > 0) mainCtx.drawImage(src.canvas, 0, 0, mainCanvas.width, mainCanvas.height);
    }
  }

  if (handLandmarks && layers['hand-default'] && layers['hand-horns']) {
    const src = isCornas() ? layers['hand-horns'] : layers['hand-default'];
    if (src.canvas.width > 0) mainCtx.drawImage(src.canvas, 0, 0, mainCanvas.width, mainCanvas.height);
  }
}

function isEyeOpen(leftIdx, rightIdx) {
  if (!faceLandmarks) return true;
  const lm = faceLandmarks;
  const p1 = lm[leftIdx], p2 = lm[rightIdx];
  if (!p1 || !p2) return true;
  return Math.abs(p1.y - p2.y) > 0.02;
}

function isMouthOpen() {
  if (!faceLandmarks) return false;
  const lm = faceLandmarks;
  const p1 = lm[13], p2 = lm[14];
  if (!p1 || !p2) return false;
  return Math.abs(p1.y - p2.y) > 0.03;
}

function isCornas() {
  if (!handLandmarks) return false;
  const lm = handLandmarks;
  return lm[8].y < lm[6].y &&
         lm[12].y > lm[10].y &&
         lm[16].y > lm[14].y &&
         lm[20].y > lm[18].y;
}

init();