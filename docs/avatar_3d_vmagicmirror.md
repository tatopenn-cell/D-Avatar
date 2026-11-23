# D-Avatar → VMagicMirror (VMC / OSC)

Guida per collegare D-Avatar a VMagicMirror come **ricevitore VMC** (solo in ricezione, l'avatar 3D è disegnato da VMagicMirror). D-Avatar non tocca la camera di VMagicMirror: la scheda "Face Tracker" resta **spenta**.

## 1. Scaricare VMagicMirror

VMagicMirror è open source (MIT): https://github.com/malaybaku/VMagicMirror

Scaricare lo zip dell'ultima release dalla pagina "Releases" del repository. Non è necessario compilarlo né scaricare eseguibili separati: l'utente estrae lo zip e avvia `VMagicMirror.exe` a mano.

## 2. Attivare il protocollo VMC in VMagicMirror

1. Aprire VMagicMirror.
2. Aprire la finestra **Impostazioni** (Setting window).
3. Selezionare la scheda **VMCP**.
4. Spuntare **"Show VMCP Tab on Main Window"** per rendere visibile il pannello VMC nella finestra principale.
5. Nella scheda **VMC Protocol** che appare:
   - Spuntare **"Enable VMC Protocol"** (abilita la ricezione).
   - In **Data to apply**, selezionare:
     - **Upper Body** (serve per la rotazione della testa)
     - **Facial (BlendShape)** (serve per occhi, bocca, sopracciglia)
   - Impostare la **Port number** (es. `39539`, la default di D-Avatar).
   - Cliccare **Apply**.
6. L'area **Status** deve mostrare un segno di spunta quando D-Avatar invia dati.

## 3. Avatar VRM da usare

Caricare un file **.vrm** (VRM 0.x o VRM 1.0) in VMagicMirror. I blendshape usati da D-Avatar sono i preset VRM standard:
- `jawOpen` → apertura bocca (`mouth_open` in D-Avatar)
- `eyeBlinkLeft` / `eyeBlinkRight` → chiusura occhi (`1 - eye_l`, `1 - eye_r`; entrambi `1.0` quando `blink = 1`)
- `Head` osso → rotazione testa (`head_yaw`, `head_pitch`, `head_roll`)

Altri blendshape VRM (es. `mouthA`, `mouthI`, `mouthU`, `mouthE`, `mouthO` per il lip-sync) non sono usati dal percorso predefinito di D-Avatar.

## 4. Schema dei messaggi OSC inviati da D-Avatar

D-Avatar invia su UDP (localhost, porta configurabile, default `39539`):

| Indirizzo OSC | Tipo argomenti | Significato |
|---|---|---|
| `/VMC/Ext/Blend/Val` | `string, float` | Nome blendshape + valore (uno per blendshape attivo) |
| `/VMC/Ext/Blend/Apply` | (nessuno) | Applica i blendshape accumulati (una volta per frame) |
| `/VMC/Ext/Bone/Pos` | `string, float×7` | `"Head"`, posX/Y/Z = `0`, rotX/Y/Z/W = quaternione dagli angoli di testa |

Esempio per un frame con `mouth_open=0.5`, `eye_l=1.0`, `eye_r=0.0`, `head_yaw=90°`:

```
/VMC/Ext/Blend/Val  "jawOpen"  0.5
/VMC/Ext/Blend/Val  "eyeBlinkLeft"  0.0
/VMC/Ext/Blend/Val  "eyeBlinkRight"  1.0
/VMC/Ext/Blend/Apply
/VMC/Ext/Bone/Pos  "Head"  0.0 0.0 0.0  0.0 0.0 0.7071 0.7071
```

Il quaternione è nell'ordine Unity (x, y, z, w), calcolato dagli angoli di Eulero yaw/pitch/roll.

## 5. Cosa resta spento

- La scheda **"Face Tracker"** di VMagicMirror resta **SPENTA**: D-Avatar non usa MediaPipe, dlib, Barracuda né nessun modello pre-addestrato per il tracciamento del volto.
- VMagicMirror si limita a disegnare l'avatar 3D (VRM) ricevendo i parametri via VMC/OSC.

## 6. Corrispondenza parametri D-Avatar → VMagicMirror

| Parametro D-Avatar | Blendshape VRM / osso | Range |
|---|---|---|
| `mouth_open` | `jawOpen` | [0, 1] |
| `eye_l` | `eyeBlinkLeft` = `1 - eye_l` | [0, 1] |
| `eye_r` | `eyeBlinkRight` = `1 - eye_r` | [0, 1] |
| `blink` | `eyeBlinkLeft` + `eyeBlinkRight` = `1.0` quando attivo | 0 o 1 |
| `head_yaw` | osso `Head`, rotazione attorno a Z | gradi |
| `head_pitch` | osso `Head`, rotazione attorno a Y | gradi |
| `head_roll` | osso `Head`, rotazione attorno a X | gradi |

## 7. Avviare l'app D-Avatar

```
python examples/app.py
```

Nella finestra dell'app, impostare l'host VMC (`127.0.0.1`) e la porta (stessa di VMagicMirror, default `39539`), poi cliccare **Avvia**. VMagicMirror deve mostrare il segno di spunta nello status VMC e l'avatar 3D deve seguire i movimenti della testa e delle espressioni.
