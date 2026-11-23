# Letture Passo A — D-Avatar DA4

Questa pagina raccoglie gli output del Passo A del task DA4: lettura della documentazione Dense-Armor, verifica delle primitive di visione, verifica del protocollo VMC di VMagicMirror.

## 1. Documenti letti

| Documento | Contenuto utile per il tracciamento |
|---|---|
| `Dense-Armor\ARCHITECTURE.md` | Mappa della libreria: `roles`, `core` (vision, timeseries, drift, ecc.), `armor.py` (shield), `checks`, `bridges`. La vision è in `dense_armor.core.signals.vision`. |
| `Dense-Armor\docs\vision\index.md` | Mappa delle 9 pagine di vision; fonti bibliografiche (Dalal & Triggs 2005 HOG, Lucas & Kanade 1981, Horn & Schunck 1981, Ziani 2025, Guillot et al. 2026, Balsubramani et al. 2013). |
| `Dense-Armor\docs\vision\frames.md` | `Frame` (array + timestamp), `ArrayStream`, `CameraStream`, `ImageFolderStream`, `VideoFileStream`, `CubeStream`. |
| `Dense-Armor\docs\vision\features.md` | `FrameFeatures`: HOG per celle, momenti di intensità (mean, contrast, centroid_x, centroid_y), colore (mean_R/G/B, contrast_R/G/B). |
| `Dense-Armor\docs\vision\flow.md` | Lucas–Kanade per cella con equazione `I_x u + I_y v = -I_t`, piramide per moti grandi, `DenseFlow` (Horn–Schunck), `MotionMask` (pixels che si muovono rispetto al moto globale). |
| `Dense-Armor\docs\vision\reduce.md` | `RandomProjection` (Lemma di Johnson–Lindenstrauss), `IncrementalPCA` (regola di Oja), catena `FrameFeatures | IncrementalPCA | OnlineRobustMahalanobis`. |
| `Dense-Armor\docs\vision\motion.md` | `MotionEnergy` (cambiamento per frame), `GlobalMotion` (shift globale con phase correlation), `SceneChange`, `FrameAnomaly`, `FrameWindow` (ring buffer con media streaming, std, motion energy, quantili DDSketch). |
| `Dense-Armor\docs\vision\tracking.md` | `MovingObjectTracker` (background robusto, componenti connessi, Kalman, associazione due round), `TrackToSignal`, `FeatureToSignal`, `FrameJointFusion`. |
| `Dense-Armor\docs\vision\kinematics.md` | `LandmarkStream`, `LandmarkCleaner`, `Kinematics`, `Registration` (shift con phase correlation), `Stabilizer` (rimuove il moto globale), `DifferenceImage`. |
| `Dense-Armor\docs\vision\quality.md` | `FrameQuality` (sharpness, exposure, noise, flag black frame / frame ripetuto), `Sync` (cross-correlazione offset orologio), `EventFrame`. |
| `Dense-Armor\docs\vision\patches.md` | `PatchFeatures` (descrittori per patch 16x16, stride 8, 9 bin HOG + mean + std), `PatchMemory` (memoria reservoir/coreset, PatchCore eq. 6). |
| `Dense-Armor\docs\timeseries\index.md` + `kalman.md`, `ar.md`, `garch.md`, `cycle.md` | KalmanFilter, OnlineAR, GARCH11, CycleResidual — usabili per filtrare i segnali dei parametri. |
| `Dense-Armor\docs\drift\index.md` + `detectors.md`, `cusum.md` | CUSUMDriftDetector, ADWIN, KSWIN, PageHinkley — drift detection su segnali scalari. |

## 2. Verifica import primitive candidate

Ogni primitiva nominata nel task importa senza errori dal modulo indicato:

| Primitiva | Modulo | Esito |
|---|---|---|
| `DenseFlow` | `dense_armor.core.signals.vision.dense` | OK |
| `GlobalMotion` | `dense_armor.core.signals.vision.motion` | OK |
| `MotionEnergy` | `dense_armor.core.signals.vision.motion` | OK |
| `MovingObjectTracker` | `dense_armor.core.signals.vision.tracking` | OK |
| `Registration` | `dense_armor.core.signals.vision.registration` | OK |
| `Stabilizer` | `dense_armor.core.signals.vision.registration` | OK |
| `FrameWindow` | `dense_armor.core.signals.vision.window` | OK |
| `FrameNormalizer` | `dense_armor.core.signals.vision.robustness` | OK |
| `PatchFeatures` | `dense_armor.core.signals.vision` | OK |
| `PatchMemory` | `dense_armor.core.signals.vision` | OK |
| `FrameQuality` | `dense_armor.core.signals.vision.quality` | OK |
| `TrackToSignal` | `dense_armor.core.signals.vision.adapters` | OK |
| `FeatureToSignal` | `dense_armor.core.signals.vision.adapters` | OK |
| `Kinematics` | `dense_armor.core.signals.vision` | OK |
| `FrameJointFusion` | `dense_armor.core.signals.vision.fusion` | OK |

Verifica eseguita con `python -c "from <modulo> import <Nome>"` per ogni riga sopra: nessuna eccezione.

## 3. Output `launch.py select`

Il comando `python Dense-Armor\dense_armor\usable\launch.py select "<lavoro>"` è stato provato con i nove lavori indicati nel task (trovare la regione del volto senza modello; movimento globale della testa; inclinazione roll/yaw/pitch; apertura bocca; apertura occhi; ammiccamento; stabilizzare video; sincronizzare video e parametri). Il sistema `usable` accetta solo le `NEEDS` in inglese (`classify`, `regress`, `detect_drift`, `detect_anomaly`, `forecast`, `audio_features`, `robot`, `finance_controls`, `summarize_stream`, `text_features`, `vision_features`, `resources`, `cluster`, `heal`): nessuno dei nove lavori italiani corrisponde a una `NEED` valida, quindi `select` restituisce output vuoto per tutti. `python Dense-Armor\dense_armor\usable\launch.py list --domain vision` restituisce ugualmente output vuoto (nessuna primitive matching). Questo è l'output vero del comando su questo ambiente.

## 4. VMagicMirror — lato ricezione VMC (dati dal codice C#)

### Messaggi OSC accettati (da `VMCPDataDefinition.cs`)

| Indirizzo OSC | `VMCPMessageType` | Valori dell'argomento (ordine) |
|---|---|---|
| `/VMC/Ext/Root/Pos` | `DefineRoot` | pose radice (pos + rot) |
| `/VMC/Ext/Bone/Pos` | `ForwardKinematics` | `string nomeOsso`, `float posX`, `posY`, `posZ`, `float rotX`, `rotY`, `rotZ`, `rotW` |
| `/VMC/Ext/Tra/Pos` | `TrackerPose` | `string nomeTracker`, `float posX`, `posY`, `posZ`, `float rotX`, `rotY`, `rotZ`, `rotW` |
| `/VMC/Ext/Blend/Val` | `BlendShapeValue` | `string chiaveBlendShape`, `float valore` |
| `/VMC/Ext/Blend/Apply` | `BlendShapeApply` | (applica i valori accumulati) |

Fonte: `VMagicMirror\Assets\Baku\VMagicMirror\Scripts\InputMonitoring\VMCP\VMCPDataDefinition.cs`, righe 33-37 (switch su `message.address`) e righe 75-94 (`GetPose` legge `posX,posY,posZ,rotX,rotY,rotZ,rotW`).

### Ordine del quaternione per la testa

L'osso testa si muove con `/VMC/Ext/Bone/Pos` passando `"Head"` come primo valore (stringa), poi 3 float di posizione e 4 float di rotazione. Il costruttore Unity è `new Quaternion(rotX, rotY, rotZ, rotW)` (ordine x, y, z, w) — `VMCPDataDefinition.cs`, riga 93.

### Nomi blendshape (da `VMCPBlendShapeKeys.cs` e `VMCPBlendShape.cs`)

Chiavi accettate (camelCase, da `VMCPBlendShapePerfectSyncKeys.TrySet`):
- Occhi: `eyeBlinkLeft`, `eyeBlinkRight`, `eyeLookUpLeft/Right`, `eyeLookDownLeft/Right`, `eyeLookInLeft/Right`, `eyeLookOutLeft/Right`, `eyeWideLeft/Right`, `eyeSquintLeft/Right`
- Bocca: `mouthClose`, `mouthFunnel`, `mouthPucker`, `mouthLeft/Right`, `mouthSmileLeft/Right`, `mouthFrownLeft/Right`, `mouthPressLeft/Right`, `mouthUpperUpLeft/Right`, `mouthLowerDownLeft/Right`, `mouthStretchLeft/Right`, `mouthDimpleLeft/Right`, `mouthShrugUpper`, `mouthShrugLower`, `mouthRollUpper`, `mouthRollLower`
- Mascella: `jawOpen`, `jawForward`, `jawLeft`, `jawRight`
- Naso: `noseSneerLeft/Right`
- Guance: `cheekPuff`, `cheekSquintLeft/Right`
- Sopracciglia: `browDownLeft/Right`, `browOuterUpLeft/Right`, `browInnerUp`
- Lingua: `tongueOut`

Blendshape VRM standard per il parlato (da `VMCPBlendShape.cs`, `LipSyncKeys`): `ExpressionKey.Aa`, `Ee`, `Ih`, `Oh`, `Ou` (corrispondono a VRM A, I, U, E, O). `mouthClose` / `mouthOpen` mappano su `jawOpen` per l'apertura della bocca.

Fonte: `VMagicMirror\Assets\Baku\VMagicMirror\Scripts\InputMonitoring\VMCP\VMCPBlendShapeKeys.cs` (righe 7-98, 101-171) e `VMCPBlendShape.cs` (righe 18-25, 93-111).

### Porta default

- Porta di invio (VMagicMirror → esterno): `9000` (`SendPort = 9000` in `VMCPSendSetting.cs` riga 7 e `VMCPSender.cs` riga 410).
- Porta di ricezione (esterno → VMagicMirror): configurabile dall'utente nel pannello VMC Protocol (`source.Port` in `VMCPSources.cs` riga 21, impostata in `VMCPReceiver.cs` riga 273 `server.port = source.Port`). Nessuna costante fissa nel codice: l'utente sceglie la porta.

### Cosa spuntare nelle impostazioni VMagicMirror

Da `docs\langs\_english\docs\vmc_protocol.md` (righe 32-67):
1. Impostazioni > scheda `VMCP` > spunta `Show VMCP Tab on Main Window` (per mostrare l'UI nel pannello principale).
2. Nella scheda `VMC Protocol` > spunta `Enable VMC Protocol` per abilitare la ricezione.
3. `Data to apply`: seleziona `Upper Body` (per la rotazione testa) e `Facial (BlendShape)` (per le espressioni).
4. Imposta la porta di ricezione.
5. La scheda `Face Tracker` resta **SPENTA** (nessun modello MediaPipe/dlib/onnx nel percorso predefinito).

Fonte: `VMagicMirror\docs\langs\_english\docs\vmc_protocol.md` (righe 32-67), `VMCPReceiver.cs` (righe 190-246 per l'attivazione di blendshape e head pose).

### Riepilogo per il mapper VMC (Passo E)

| Parametro D-Avatar | Messaggio OSC VMC | Valore |
|---|---|---|
| `mouth_open` | `/VMC/Ext/Blend/Val` | chiave `jawOpen` (o `mouthA`), valore 0..1 |
| `eye_l` / `eye_r` | `/VMC/Ext/Blend/Val` | chiave `eyeBlinkLeft` / `eyeBlinkRight`, valore 1 - apertura (0..1) |
| `blink` | `/VMC/Ext/Blend/Val` | chiave `eyeBlinkLeft` + `eyeBlinkRight`, valore 0 o 1 |
| `head_yaw` / `head_pitch` / `head_roll` | `/VMC/Ext/Bone/Pos` | osso `Head`, quaternione (x, y, z, w) dagli angoli di Eulero |

Queste fonti vanno citate nel Passo E (`docs\avatar_3d_vmagicmirror.md`).

