"""
serve.py
--------
A minimal REST inference API for a trained DAS forest. This is a SEPARATE
Flask app from app.py (which is the benchmark visualizer on port 5050) —
this one just answers /predict requests against whatever forest was last
saved by demo_torch.py.

It does NOT train anything. If checkpoints/forest/ doesn't exist yet, it
tells you to go run demo_torch.py first and exits — there is no fallback
"random untrained forest" mode, because serving random noise and calling
it a prediction would be dishonest.

Run:
    conda run -n das python serve.py

Example:
    curl http://localhost:5060/health
    # -> {"status": "ok", "leaves": 3}

    python3 -c "
import json, numpy as np, urllib.request
pixels = np.zeros(784).tolist()   # replace with a real flattened MNIST image
req = urllib.request.Request('http://localhost:5060/predict',
        data=json.dumps({'pixels': pixels}).encode(),
        headers={'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read())
"
    # or: curl -X POST http://localhost:5060/predict -H 'Content-Type: application/json' -d '{"pixels": [...784 floats...]}'
"""
import json
import os
import sys

import torch
from flask import Flask, jsonify, request

sys.path.insert(0, '.')
from das_torch import load_forest

PORT = 5060
DEVICE = "cpu"
FOREST_DIR = './checkpoints/forest'

app = Flask(__name__)

if not os.path.isdir(FOREST_DIR):
    print(f"\n  No saved forest found at {FOREST_DIR}/.")
    print(f"  Run `conda run -n das python demo_torch.py` first to train and checkpoint a forest.\n")
    sys.exit(1)

print(f"Loading forest from {FOREST_DIR} ...")
forest = load_forest(FOREST_DIR, device=DEVICE)
forest.eval()
n_leaves = len(forest.leaves)
expected_dim = forest.router.gate.in_features
print(f"Loaded forest: {n_leaves} leaves, expects {expected_dim}-dim flat input.")

# Optional sidecar from demo_torch.py: maps leaf id -> (neg_digit, pos_digit)
# so a leaf's binary 0/1 output can be translated back into an actual MNIST
# digit. If it's missing (e.g. a forest saved by some other script), we just
# report the raw binary class instead of a digit — no guessing.
domain_labels = None
labels_path = os.path.join(FOREST_DIR, 'domain_labels.json')
if os.path.exists(labels_path):
    with open(labels_path) as f:
        raw = json.load(f)
    domain_labels = {int(k): tuple(v) for k, v in raw.items()}
    print(f"Loaded domain labels: {domain_labels}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "leaves": n_leaves})

@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json(silent=True)
    if body is None or 'pixels' not in body:
        return jsonify({"error": "expected JSON body {\"pixels\": [...]}"}), 400

    pixels = body['pixels']
    if not isinstance(pixels, list) or len(pixels) != expected_dim:
        return jsonify({
            "error": f"'pixels' must be a list of {expected_dim} floats, got "
                     f"{'a ' + type(pixels).__name__ if not isinstance(pixels, list) else f'length {len(pixels)}'}"
        }), 400
    try:
        x = torch.tensor([float(p) for p in pixels], dtype=torch.float32).unsqueeze(0)
    except (TypeError, ValueError):
        return jsonify({"error": "'pixels' must all be numbers"}), 400

    with torch.no_grad():
        out, leaf_idx = forest.predict(x)
        probs = torch.softmax(out, dim=-1)
        binary_class = probs.argmax(dim=-1).item()
        confidence = probs.max(dim=-1).values.item()
    leaf = leaf_idx.item()

    if domain_labels is not None and leaf in domain_labels:
        neg_digit, pos_digit = domain_labels[leaf]
        prediction = pos_digit if binary_class == 1 else neg_digit
    else:
        prediction = binary_class  # no digit mapping available; report raw class

    return jsonify({"leaf": leaf, "prediction": prediction, "confidence": confidence})

if __name__ == "__main__":
    print(f"\n  -> DAS inference API: http://localhost:{PORT}")
    print(f"  -> GET  /health")
    print(f"  -> POST /predict   {{\"pixels\": [784 floats]}}\n")
    app.run(debug=False, port=PORT, threaded=True)
