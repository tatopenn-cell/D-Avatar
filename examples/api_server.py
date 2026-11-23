"""API server: espone le primitive Dense-Armor al frontend HTML/JS via HTTP."""
from __future__ import annotations

from flask import Flask, request, jsonify
from dense_armor.core.signals.vision.features import FrameFeatures
from dense_armor.core.signals.vision.motion import GlobalMotion, MotionEnergy
from dense_armor.core.signals.vision.registration import Registration, Stabilizer
from dense_armor.core.signals.vision.tracking import MovingObjectTracker
from dense_armor.core.signals.vision.landmarks import LandmarkStream, BatchedLandmarkShield
from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.roles import RealtimePipeline, Transformer
from dense_armor._dtype import as_default_array
import numpy as np

app = Flask(__name__)

# Istanzia una volta le primitive (stato persistente tra le richieste)
features = FrameFeatures()
global_motion = GlobalMotion()
motion_energy = MotionEnergy()
registration = Registration()
stabilizer = Stabilizer()
moving_tracker = MovingObjectTracker()
landmark_stream = LandmarkStream()
shield = BatchedLandmarkShield()
adwin = ADWIN()

@app.post("/features/transform")
def features_transform():
    data = request.get_json(force=True)
    arr = np.array(data["array"], dtype=np.float32)
    result = features.transform_one(arr)
    return jsonify(result)

@app.post("/motion/global")
def motion_global():
    data = request.get_json(force=True)
    arr = np.array(data["array"], dtype=np.float32)
    result = global_motion.transform_one(arr)
    return jsonify(result)

@app.post("/motion/energy")
def motion_energy_endpoint():
    data = request.get_json(force=True)
    arr = np.array(data["array"], dtype=np.float32)
    result = motion_energy.transform_one(arr)
    return jsonify(result)

@app.post("/registration/transform")
def registration_transform():
    data = request.get_json(force=True)
    arr = np.array(data["array"], dtype=np.float32)
    result = registration.transform_one(arr)
    return jsonify(result)

@app.post("/stabilizer/transform")
def stabilizer_transform():
    data = request.get_json(force=True)
    arr = np.array(data["array"], dtype=np.float32)
    result = stabilizer.transform_one(arr)
    return jsonify(result)

@app.post("/tracker/moving")
def tracker_moving():
    data = request.get_json(force=True)
    arr = np.array(data["array"], dtype=np.float32)
    result = moving_tracker.transform_one(arr)
    return jsonify(result)

@app.post("/shield/transform")
def shield_transform():
    data = request.get_json(force=True)
    pts = as_default_array(np.array(data["points"], dtype=np.float32))
    conf = as_default_array(np.array(data.get("confidence", np.ones(len(pts))), dtype=np.float32))
    t = data.get("t")
    result = shield.transform_one({"points": pts, "confidence": conf, "t": t})
    return jsonify({
        "points": result["points"].tolist() if hasattr(result["points"], "tolist") else result["points"],
        "cleaned": result["cleaned"].tolist() if hasattr(result["cleaned"], "tolist") else result["cleaned"],
    })

@app.post("/adwin/update")
def adwin_update():
    data = request.get_json(force=True)
    val = float(data["value"])
    adwin.update(val)
    return jsonify({"drift_detected": bool(adwin.drift_detected)})

@app.post("/pipeline/build")
def pipeline_build():
    data = request.get_json(force=True)
    period_s = float(data.get("period_s", 0.05))
    max_step = float(data.get("max_step", 0.1))
    max_step_deg = float(data.get("max_step_deg", 5.0))
    pipeline = RealtimePipeline(period_s=period_s)
    return jsonify({"status": "built", "period_s": period_s})

@app.get("/status")
def status():
    return jsonify({"status": "ok", "primitives": [
        "FrameFeatures", "GlobalMotion", "MotionEnergy", "Registration",
        "Stabilizer", "MovingObjectTracker", "LandmarkStream",
        "BatchedLandmarkShield", "ADWIN", "RealtimePipeline"
    ]})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
