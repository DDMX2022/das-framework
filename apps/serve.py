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

# ── Landing page + sample endpoint (so the API isn't a blank page) ──────
def _load_test_images():
    """Lazily load MNIST test images for the interactive tester. Returns None
    if the IDX files aren't present (the tester just hides itself then)."""
    import gzip, struct
    try:
        with gzip.open('./data/MNIST/raw/t10k-images-idx3-ubyte.gz', 'rb') as f:
            f.read(4); n = struct.unpack('>I', f.read(4))[0]; f.read(8)
            data = __import__('numpy').frombuffer(f.read(), dtype='uint8').reshape(n, 784)
        with gzip.open('./data/MNIST/raw/t10k-labels-idx1-ubyte.gz', 'rb') as f:
            f.read(8)
            labels = __import__('numpy').frombuffer(f.read(), dtype='uint8')
        return data, labels
    except Exception:
        return None

@app.route('/sample', methods=['GET'])
def sample():
    import random
    imgs = _load_test_images()
    if imgs is None:
        return jsonify({"error": "no MNIST test data on disk"}), 404
    data, labels = imgs
    i = random.randrange(len(data))
    return jsonify({"pixels": (data[i] / 255.0).tolist(), "label": int(labels[i])})

@app.route('/', methods=['GET'])
def index():
    return """<!doctype html><html><head><meta charset=utf-8>
<title>DAS inference API</title><style>
body{background:#0a140a;color:#c8e6c9;font-family:'SF Mono',monospace;max-width:680px;margin:40px auto;padding:0 16px;line-height:1.6}
h1{color:#68d391;font-size:1.1rem;letter-spacing:.08em}
.card{background:#0f1c0f;border:1px solid #1a3020;border-radius:8px;padding:16px;margin:14px 0}
button{background:#4a7c59;color:#fff;border:none;border-radius:5px;padding:8px 16px;font-family:inherit;cursor:pointer}
button:hover{background:#56ab2f}
canvas{image-rendering:pixelated;border:1px solid #1a3020;background:#000}
code{color:#9ae6b4}.muted{color:#4a7c59;font-size:.85rem}
#out{margin-top:10px;font-size:.95rem}
</style></head><body>
<h1>🌳 DAS inference API</h1>
<p class=muted>A trained forest is loaded and serving predictions. This is a JSON API; the tester below calls it for you.</p>
<div class=card>
  <button onclick=go()>Classify a random MNIST digit</button>
  <div style=margin-top:12px><canvas id=c width=112 height=112></canvas></div>
  <div id=out class=muted>click the button…</div>
</div>
<div class=card>
  <div class=muted>Endpoints</div>
  <p><code>GET /health</code> — forest status<br>
     <code>GET /sample</code> — a random test image + label<br>
     <code>POST /predict</code> — <code>{"pixels":[784 floats]}</code> → <code>{leaf,prediction,confidence}</code></p>
</div>
<script>
async function go(){
  const s=await (await fetch('/sample')).json();
  if(s.error){document.getElementById('out').textContent=s.error;return;}
  const ctx=document.getElementById('c').getContext('2d');
  const img=ctx.createImageData(28,28);
  for(let i=0;i<784;i++){const v=Math.round(s.pixels[i]*255);img.data[i*4]=v;img.data[i*4+1]=v;img.data[i*4+2]=v;img.data[i*4+3]=255;}
  const tmp=document.createElement('canvas');tmp.width=28;tmp.height=28;tmp.getContext('2d').putImageData(img,0,0);
  ctx.imageSmoothingEnabled=false;ctx.clearRect(0,0,112,112);ctx.drawImage(tmp,0,0,112,112);
  const r=await (await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pixels:s.pixels})})).json();
  const ok=r.prediction===s.label;
  document.getElementById('out').innerHTML=
    `true label <b>${s.label}</b> → routed to <b>leaf ${r.leaf}</b>, predicted <b>${r.prediction}</b> `+
    `(${(r.confidence*100).toFixed(1)}%) <span style="color:${ok?'#68d391':'#fc8181'}">${ok?'✓':'✗ (out-of-domain digits misroute by design)'}</span>`;
}
</script></body></html>"""

if __name__ == "__main__":
    print(f"\n  -> DAS inference API: http://localhost:{PORT}")
    print(f"  -> GET  /health")
    print(f"  -> POST /predict   {{\"pixels\": [784 floats]}}\n")
    app.run(debug=False, port=PORT, threaded=True)
