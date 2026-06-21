"""
app.py — DAS visualiser server
  /        full forest UI with Fibonacci spiral growth
  /train   SSE stream for demo training
  /benchmark SSE stream for real-data benchmark
"""
import json, sys
import numpy as np
from flask import Flask, render_template, Response

sys.path.insert(0, '.')
from das.model import DASForest
from das.functional import FibonacciLeaf, softmax
from das.routing import StemRouter
from das.packnet import PackNetMLP

app = Flask(__name__)

def ce_grad(logits, y):
    p = softmax(logits); oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return (p - oh) / len(y)

def acc(logits, y):
    return round(float((logits.argmax(1) == y).mean()), 4)

def emit(d):
    return f"data: {json.dumps(d)}\n\n"

def n_params(leaf):
    return sum(w.size for w in leaf.W) + sum(b.size for b in leaf.b)

# ── Phase 1: Synthetic demo ────────────────────────────────────
D_MODEL, LEAF_DIMS, N = 21, [21, 13, 8, 2], 600

def make_domain(did, n, rng):
    centers = {0: np.eye(D_MODEL)[0]*4, 1: np.eye(D_MODEL)[7]*4, 2: np.eye(D_MODEL)[14]*4}
    rule = np.random.default_rng(100+did).normal(0, 1, D_MODEL)
    X = centers[did] + rng.normal(0, 1.0, (n, D_MODEL))
    return X, (X @ rule > 0).astype(int), np.full(n, did)

def run_training():
    rng = np.random.default_rng(42)
    X0,y0,d0 = make_domain(0, N, rng)
    X1,y1,d1 = make_domain(1, N, rng)
    Xr = np.vstack([X0,X1]); dr = np.concatenate([d0,d1])
    forest = DASForest(D_MODEL, LEAF_DIMS, num_leaves=2, seed=7)

    yield emit({'e':'phase','n':1,'label':'Roots anchor — Stem Router learns domain boundaries'})
    for s in range(600):
        idx = rng.integers(0,len(Xr),128)
        loss,a = forest.router.train_step(Xr[idx], dr[idx], lr=0.1)
        if s%20==0: yield emit({'e':'router','step':s,'acc':round(a,4),'loss':round(float(loss),4)})
    ridx,_ = forest.router.route(Xr)
    yield emit({'e':'phase_done','n':1,'acc':round(float((ridx==dr).mean()),4)})

    for lid,(X,y,Xd) in enumerate([(X0,y0,d0),(X1,y1,d1)]):
        n_phase = lid+2
        yield emit({'e':'phase','n':n_phase,'leaf':lid,'label':f'Spiral {lid} unfurls — Leaf {lid} trains in isolation'})
        leaf = forest.leaves[lid]; leaf.frozen = False
        for s in range(400):
            idx = rng.integers(0,N,128)
            leaf.backward(ce_grad(leaf.forward(X[idx]), y[idx]), 0.05)
            if s%20==0: yield emit({'e':'leaf','leaf':lid,'step':s,'acc':acc(leaf.forward(X),y)})
        leaf.frozen = True
        yield emit({'e':'phase_done','n':n_phase,'leaf':lid,'acc':acc(leaf.forward(X),y)})

    before = forest.leaf_hashes()
    yield emit({'e':'snap','when':'before','h':{str(k):v for k,v in before.items()}})

    yield emit({'e':'phase','n':4,'leaf':2,'label':'New shoot grafts — Leaf 2 sprouts for Domain 2'})
    X2,y2,d2 = make_domain(2, N, rng)
    new_id = forest.graft(seed=321)
    Xra = np.vstack([X0,X1,X2]); dra = np.concatenate([d0,d1,d2])
    for s in range(400):
        idx = rng.integers(0,len(Xra),128)
        forest.router.train_step(Xra[idx], dra[idx], lr=0.1)
    forest.leaves[0].frozen = True; forest.leaves[1].frozen = True
    leaf = forest.leaves[new_id]; leaf.frozen = False
    for s in range(400):
        idx = rng.integers(0,N,128)
        leaf.backward(ce_grad(leaf.forward(X2[idx]), y2[idx]), 0.05)
        if s%20==0: yield emit({'e':'leaf','leaf':2,'step':s,'acc':acc(leaf.forward(X2),y2)})
    leaf.frozen = True
    yield emit({'e':'phase_done','n':4,'leaf':2,'acc':acc(leaf.forward(X2),y2)})

    after = forest.leaf_hashes()
    yield emit({'e':'snap','when':'after','h':{str(k):v for k,v in after.items()}})
    passed = all(before[k]==after[k] for k in before)
    yield emit({'e':'proof','passed':passed})
    results = []
    for X,y,name,did in [(X0,y0,'Domain 0',0),(X1,y1,'Domain 1',1),(X2,y2,'Domain 2',2)]:
        out,li = forest.predict(X)
        results.append({'name':name,'routed':round(float((li==did).mean()),3),'acc':acc(out,y)})
    yield emit({'e':'final','results':results})
    yield emit({'e':'done'})


# ── Phase 2: Benchmark ─────────────────────────────────────────
BM_LEAF = [64,32,16,2]; BM_BASE = [64,80,40,2]; BM_LR=0.02; BM_B=64

def load_digits_data():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    dg = load_digits()
    X = StandardScaler().fit_transform(dg.data.astype(np.float64))
    yd = dg.target
    DMAP={0:0,1:0,2:0,3:0,4:1,5:1,6:1,7:2,8:2,9:2}
    TASK={0:lambda d:int(d%2==0),1:lambda d:int(d<=5),2:lambda d:int(d<=8)}
    dom=np.array([DMAP[y] for y in yd])
    lab=np.array([TASK[DMAP[y]](y) for y in yd])
    splits={}
    for d in range(3):
        m=dom==d
        Xtr,Xte,ytr,yte=train_test_split(X[m],lab[m],test_size=0.2,random_state=42)
        splits[d]=(Xtr,ytr,Xte,yte)
    Xall=np.vstack([splits[d][0] for d in range(3)])
    dall=np.concatenate([np.full(len(splits[d][0]),d) for d in range(3)])
    return splits, Xall, dall

def run_benchmark():
    rng=np.random.default_rng(42)
    try: splits,Xall,dall=load_digits_data()
    except ImportError:
        yield emit({'e':'berror','msg':'Run: pip install scikit-learn'}); return

    forest=DASForest(64,BM_LEAF,num_leaves=3,seed=7)
    das_p=forest.router.W.size+forest.router.b.size+sum(n_params(l) for l in forest.leaves)

    yield emit({'e':'bphase','n':1,'label':'Router learns digit clusters (real pixels)'})
    for s in range(800):
        idx=rng.integers(0,len(Xall),BM_B)
        _,a=forest.router.train_step(Xall[idx],dall[idx],lr=0.05)
        if s%40==0: yield emit({'e':'brouter','step':s,'acc':round(a,4)})
    ridx,_=forest.router.route(Xall)
    yield emit({'e':'bphase_done','n':1,'acc':round(float((ridx==dall).mean()),4)})

    before=forest.leaf_hashes(); das_test={}
    for d in range(3):
        Xtr,ytr,Xte,yte=splits[d]
        yield emit({'e':'bphase','n':d+2,'leaf':d,'label':f'Leaf {d} — isolated on digit domain {d}'})
        leaf=forest.leaves[d]; leaf.frozen=False
        for s in range(600):
            idx=rng.integers(0,len(Xtr),BM_B)
            leaf.backward(ce_grad(leaf.forward(Xtr[idx]),ytr[idx]),BM_LR)
            if s%40==0: yield emit({'e':'bleaf','leaf':d,'step':s,'acc':acc(leaf.forward(Xtr),ytr)})
        leaf.frozen=True
        das_test[d]=acc(leaf.forward(Xte),yte)
        yield emit({'e':'bphase_done','n':d+2,'leaf':d,'test_acc':das_test[d]})
    after=forest.leaf_hashes()

    yield emit({'e':'bphase','n':5,'label':'Baseline MLP — same params, all domains mixed'})
    baseline=FibonacciLeaf(BM_BASE,seed=99); bl_p=n_params(baseline)
    Xbl=np.vstack([splits[d][0] for d in range(3)])
    ybl=np.concatenate([splits[d][1] for d in range(3)])
    for s in range(1200):
        idx=rng.integers(0,len(Xbl),BM_B)
        baseline.backward(ce_grad(baseline.forward(Xbl[idx]),ybl[idx]),BM_LR)
        if s%60==0: yield emit({'e':'bbaseline','step':s,'acc':acc(baseline.forward(Xbl),ybl)})
    bl_test={d:acc(baseline.forward(splits[d][2]),splits[d][3]) for d in range(3)}
    yield emit({'e':'bphase_done','n':5,'bl_accs':[bl_test[d] for d in range(3)]})

    passed=all(before[k]==after[k] for k in before)
    yield emit({'e':'bproof','passed':passed})
    yield emit({'e':'bresult',
                'das_accs':[das_test[d] for d in range(3)],
                'bl_accs':[bl_test[d] for d in range(3)],
                'das_params':das_p,'bl_params':bl_p})
    yield emit({'e':'bdone'})


# ── MNIST Stress: 10 leaves, 784-dim, 70k samples ─────────────
ST_LEAF = [784, 256, 64, 2]   # deep Fibonacci per leaf
ST_BASE = [784, 512, 256, 2]  # baseline same total param budget
ST_LR   = 0.008; ST_B = 256

def _read_mnist_idx(path):
    """Parse a gzipped MNIST IDX file using only stdlib + numpy."""
    import gzip, struct
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        ndims = magic & 0xFF
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(ndims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_mnist():
    base = './data/MNIST/raw'
    X_tr = _read_mnist_idx(f'{base}/train-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float64) / 255.0
    y_tr = _read_mnist_idx(f'{base}/train-labels-idx1-ubyte.gz').astype(int)
    X_te = _read_mnist_idx(f'{base}/t10k-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float64) / 255.0
    y_te = _read_mnist_idx(f'{base}/t10k-labels-idx1-ubyte.gz').astype(int)
    # Standardise (no sklearn needed)
    mu = X_tr.mean(0); sig = X_tr.std(0) + 1e-8
    X_tr = (X_tr - mu) / sig; X_te = (X_te - mu) / sig
    # Per-digit 1-vs-rest balanced splits (85/15)
    splits = {}
    rng = np.random.default_rng(42)
    for d in range(10):
        pos_tr = np.where(y_tr == d)[0]; neg_tr = np.where(y_tr != d)[0]
        neg_pick = rng.choice(neg_tr, len(pos_tr), replace=False)
        idx = np.concatenate([pos_tr, neg_pick])
        Xall = X_tr[idx]; yall = np.array([1]*len(pos_tr) + [0]*len(neg_pick))
        # shuffle then split 85/15
        perm = rng.permutation(len(Xall))
        cut = int(len(Xall) * 0.85)
        splits[d] = (Xall[perm[:cut]], yall[perm[:cut]], Xall[perm[cut:]], yall[perm[cut:]])
    Xr = np.vstack([X_tr[y_tr == d][:600] for d in range(10)])
    dr = np.concatenate([np.full(600, d) for d in range(10)])
    return splits, Xr, dr

def run_stress():
    rng = np.random.default_rng(42)
    try: splits, Xr, dr = load_mnist()
    except FileNotFoundError:
        yield emit({'e':'serror','msg':'Run mnist_stress.py once first to download MNIST data'}); return
    except Exception as ex:
        yield emit({'e':'serror','msg':str(ex)}); return

    forest = DASForest(784, ST_LEAF, num_leaves=10, seed=7)
    das_p  = forest.router.W.size + forest.router.b.size + sum(n_params(l) for l in forest.leaves)
    yield emit({'e':'sloaded', 'n':70000})
    yield emit({'e':'sparams', 'das':das_p})

    # Router: 10-way
    yield emit({'e':'sphase','n':1,'label':'Router — 10-way routing on 784-dim MNIST pixels'})
    for s in range(1000):
        idx = rng.integers(0, len(Xr), ST_B)
        _, a = forest.router.train_step(Xr[idx], dr[idx], lr=0.03)
        if s % 50 == 0: yield emit({'e':'srouter','step':s,'acc':round(a,4)})
    ridx, _ = forest.router.route(Xr)
    yield emit({'e':'sphase_done','n':1,'acc':round(float((ridx==dr).mean()),4)})

    # 10 leaves — incremental forgetting proof
    frozen_hashes = {}   # hash of leaf d right after it's frozen
    leaf_violations = set()  # leaves whose hashes were found changed during later training
    proof_ok = True
    das_test = {}
    for d in range(10):
        Xtr,ytr,Xte,yte = splits[d]
        yield emit({'e':'sphase','n':d+2,'leaf':d,'label':f'Leaf {d} — digit {d} vs rest'})
        # Snapshot all already-frozen leaves before training leaf d
        snap_before = dict(frozen_hashes)
        leaf = forest.leaves[d]; leaf.frozen = False
        for s in range(600):
            idx = rng.integers(0, len(Xtr), ST_B)
            leaf.backward(ce_grad(leaf.forward(Xtr[idx]), ytr[idx]), ST_LR)
            if s % 40 == 0: yield emit({'e':'sleaf','leaf':d,'step':s,'acc':acc(leaf.forward(Xtr[:500]),ytr[:500])})
        leaf.frozen = True
        frozen_hashes[d] = forest.leaves[d].weight_hash()
        # Verify previously-frozen leaves are byte-identical
        for prev in range(d):
            if forest.leaves[prev].weight_hash() != snap_before[prev]:
                proof_ok = False
                leaf_violations.add(prev)
        das_test[d] = acc(leaf.forward(Xte), yte)
        yield emit({'e':'sphase_done','n':d+2,'leaf':d,'test_acc':das_test[d]})
    yield emit({'e':'ssnap','when':'after','h':{str(k):v for k,v in frozen_hashes.items()}})

    # Baseline
    yield emit({'e':'sbphase','n':12,'label':'Baseline — single MLP, all 10 domains mixed'})
    baseline = FibonacciLeaf(ST_BASE, seed=99)
    bl_p = n_params(baseline)
    yield emit({'e':'sparams_bl','bl':bl_p})
    Xbl = np.vstack([splits[d][0] for d in range(10)])
    ybl = np.concatenate([splits[d][1] for d in range(10)])
    for s in range(1000):
        idx = rng.integers(0, len(Xbl), ST_B)
        baseline.backward(ce_grad(baseline.forward(Xbl[idx]), ybl[idx]), ST_LR)
        if s % 50 == 0: yield emit({'e':'sbaseline','step':s,'acc':acc(baseline.forward(Xbl[:1000]),ybl[:1000])})
    bl_test = {d: acc(baseline.forward(splits[d][2]), splits[d][3]) for d in range(10)}
    yield emit({'e':'sphase_done','n':12,'bl_accs':[bl_test[d] for d in range(10)]})

    # Per-leaf: was it byte-stable across all subsequent training rounds?
    leaf_stable = {d: d not in leaf_violations for d in range(10)}
    yield emit({'e':'sproof','passed':proof_ok,
                'leaf_stable':{str(d):v for d,v in leaf_stable.items()},
                'das_accs':[das_test[d] for d in range(10)],
                'bl_accs':[bl_test[d] for d in range(10)]})
    yield emit({'e':'sdone'})


# ── Real-World Multi-Dataset Benchmark ──────────────────────────
RB_LR = 0.015; RB_B = 128
RB_ROUTER_STEPS = 1000; RB_LEAF_STEPS = 1000; RB_BASE_STEPS = 1500

def _make_dims(n_feat, n_dom):
    """Fibonacci-descent leaf dims + matched baseline dims."""
    h = min(64, n_feat)
    leaf_d = [n_feat, h, max(8, h//2), max(4, h//4), 2]
    base_d = [n_feat, min(256, h * n_dom), min(128, h * n_dom // 2), 2]
    return leaf_d, base_d

def _rb_split(X, y, domain, n_domains, seed=42):
    rng = np.random.default_rng(seed)
    splits = {}
    for d in range(n_domains):
        mask = domain == d
        Xd, yd = X[mask], y[mask]
        n = len(Xd); cut = int(n * 0.8)
        perm = rng.permutation(n)
        splits[d] = (Xd[perm[:cut]], yd[perm[:cut]], Xd[perm[cut:]], yd[perm[cut:]])
    Xall = np.vstack([splits[d][0] for d in range(n_domains)])
    dall = np.concatenate([np.full(len(splits[d][0]), d) for d in range(n_domains)])
    return splits, Xall, dall

def _load_adult():
    import socket, pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    socket.setdefaulttimeout(90)
    ds = fetch_openml('adult', version=2, as_frame=True, parser='auto')
    y = (ds.target == '>50K').values.astype(int)
    edu = ds.frame['education-num'].values.astype(float)
    domain = np.where(edu <= 9, 0, np.where(edu <= 13, 1, 2))
    X = pd.get_dummies(ds.data).fillna(0).values.astype(np.float64)
    X = StandardScaler().fit_transform(X)
    return X, y, domain, 3, ['Low edu (≤HS)', 'Mid (some college)', 'High (degree+)']

def _load_wine():
    import socket
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    socket.setdefaulttimeout(90)
    red   = fetch_openml('wine-quality-red',   version=1, as_frame=False, parser='auto')
    white = fetch_openml('wine-quality-white', version=1, as_frame=False, parser='auto')
    Xr = red.data.astype(np.float64);   yr = (red.target.astype(float)   >= 7).astype(int)
    Xw = white.data.astype(np.float64); yw = (white.target.astype(float) >= 7).astype(int)
    X = np.vstack([Xr, Xw]); y = np.concatenate([yr, yw])
    domain = np.array([0]*len(Xr) + [1]*len(Xw))
    X = StandardScaler().fit_transform(X)
    return X, y, domain, 2, ['Red wine', 'White wine']

def _load_credit():
    import socket, pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    socket.setdefaulttimeout(90)
    ds = fetch_openml('default-of-credit-card-clients', version=1, as_frame=True, parser='auto')
    y = ds.target.astype(int).values
    cols = list(ds.frame.columns)
    edu_col = next((c for c in ['x3','EDUCATION','education'] if c in cols), cols[2])
    edu = ds.frame[edu_col].values.astype(float)
    domain = np.where(edu == 1, 2, np.where(edu == 2, 1, 0))
    X = ds.data.values.astype(np.float64)
    X = StandardScaler().fit_transform(X)
    return X, y, domain, 3, ['HS / Other', 'University', 'Grad School']

RB_DATASETS = [
    ('Adult Income',   '48K US Census · income >$50K · domains: education tier',   _load_adult),
    ('Wine Quality',   '6.5K wine samples · quality ≥7 · domains: red vs. white',  _load_wine),
    ('Credit Default', '30K credit clients · default next month · domains: edu tier', _load_credit),
]

def run_real_bench():
    import threading, queue as _queue
    rng = np.random.default_rng(42)
    yield emit({'e':'rb_init','datasets':[
        {'n':i,'name':n,'desc':d} for i,(n,d,_) in enumerate(RB_DATASETS)
    ]})

    for ds_idx, (ds_name, ds_desc, loader) in enumerate(RB_DATASETS):
        yield emit({'e':'rb_loading','n':ds_idx,'name':ds_name,'desc':ds_desc})

        # Run loader in a thread; poll every 3s and emit heartbeat ticks so
        # the browser's "server alive" indicator stays green during slow downloads.
        result_q = _queue.Queue()
        def _run(ldr=loader):
            try:    result_q.put(('ok', ldr()))
            except Exception as ex: result_q.put(('err', ex))
        threading.Thread(target=_run, daemon=True).start()

        elapsed = 0
        while True:
            try:
                status, val = result_q.get(timeout=3)
                break
            except _queue.Empty:
                elapsed += 3
                yield emit({'e':'rb_tick','n':ds_idx,'elapsed':elapsed})

        if status == 'err':
            yield emit({'e':'rb_error','n':ds_idx,'name':ds_name,'msg':str(val)}); continue
        X, y, domain, n_dom, dnames = val

        n_feat = X.shape[1]
        leaf_dims, base_dims = _make_dims(n_feat, n_dom)
        splits, Xall, dall = _rb_split(X, y, domain, n_dom)

        yield emit({'e':'rb_loaded','n':ds_idx,'n_samples':len(X),'n_features':n_feat,
                    'n_domains':n_dom,'domain_names':dnames,
                    'leaf_dims':leaf_dims,'base_dims':base_dims})

        forest = DASForest(n_feat, leaf_dims, num_leaves=n_dom, seed=7)
        das_p = (forest.router.W.size + forest.router.b.size +
                 sum(n_params(l) for l in forest.leaves))

        # Router
        yield emit({'e':'rb_phase','n':ds_idx,'phase':'router'})
        for s in range(RB_ROUTER_STEPS):
            idx = rng.integers(0, len(Xall), RB_B)
            _, a = forest.router.train_step(Xall[idx], dall[idx], lr=RB_LR)
            if s % 100 == 0:
                yield emit({'e':'rb_router','n':ds_idx,'step':s,'acc':round(a,4)})
        ridx, _ = forest.router.route(Xall)
        racc = float((ridx == dall).mean())
        yield emit({'e':'rb_router_done','n':ds_idx,'acc':round(racc,4)})

        # Leaves in isolation
        before_h = forest.leaf_hashes()
        das_test = {}
        for d in range(n_dom):
            Xtr, ytr, Xte, yte = splits[d]
            yield emit({'e':'rb_phase','n':ds_idx,'phase':'leaf','leaf':d,'label':dnames[d]})
            leaf = forest.leaves[d]; leaf.frozen = False
            for s in range(RB_LEAF_STEPS):
                idx = rng.integers(0, len(Xtr), RB_B)
                leaf.backward(ce_grad(leaf.forward(Xtr[idx]), ytr[idx]), RB_LR)
                if s % 100 == 0:
                    yield emit({'e':'rb_leaf','n':ds_idx,'leaf':d,'step':s,
                                'acc':acc(leaf.forward(Xtr[:400]), ytr[:400])})
            leaf.frozen = True
            das_test[d] = round(acc(leaf.forward(Xte), yte), 4)
            yield emit({'e':'rb_leaf_done','n':ds_idx,'leaf':d,'test_acc':das_test[d]})
        after_h = forest.leaf_hashes()
        proof = all(before_h[k] == after_h[k] for k in before_h)

        # Baseline MLP
        yield emit({'e':'rb_phase','n':ds_idx,'phase':'baseline'})
        baseline = FibonacciLeaf(base_dims, seed=99); bl_p = n_params(baseline)
        Xbl = np.vstack([splits[d][0] for d in range(n_dom)])
        ybl = np.concatenate([splits[d][1] for d in range(n_dom)])
        for s in range(RB_BASE_STEPS):
            idx = rng.integers(0, len(Xbl), RB_B)
            baseline.backward(ce_grad(baseline.forward(Xbl[idx]), ybl[idx]), RB_LR)
            if s % 150 == 0:
                yield emit({'e':'rb_baseline','n':ds_idx,'step':s,
                            'acc':acc(baseline.forward(Xbl[:500]), ybl[:500])})
        bl_test = {d: round(acc(baseline.forward(splits[d][2]), splits[d][3]), 4)
                   for d in range(n_dom)}

        yield emit({'e':'rb_result','n':ds_idx,'name':ds_name,'desc':ds_desc,
                    'n_samples':len(X),'n_features':n_feat,
                    'n_domains':n_dom,'domain_names':dnames,
                    'das_accs':[das_test[d] for d in range(n_dom)],
                    'bl_accs':[bl_test[d] for d in range(n_dom)],
                    'das_params':das_p,'bl_params':bl_p,
                    'proof':proof,'router_acc':round(racc,4)})

    yield emit({'e':'rb_done'})


# ── Phase 5: Split-MNIST Continual Learning ─────────────────────
CL_LEAF      = [784, 64, 32, 2]    # binary leaf per task
CL_LEAF_10   = [784, 128, 64, 10]  # 10-class for multi-task upper bound
CL_LR        = 0.012; CL_B = 128
CL_ROUTER_STEPS  = 800
CL_LEAF_STEPS    = 600
CL_FT_STEPS      = 600
CL_MT_STEPS      = 1000
CL_EWC_LAMBDA    = 5000.0  # EWC penalty strength. NB: single-head Split-MNIST is the
                           # known-hard regime where EWC only partly helps (van de Ven
                           # & Tolias 2019) — it beats naive fine-tuning but cannot reach
                           # DAS's structural BWT≈0. That contrast is the point.
CL_PKN_STEPS     = 600   # training steps on free weights, per task
CL_PKN_REFIT     = 300   # brief re-finetune steps on task-owned weights after pruning
CL_TASKS = [(0,1),(2,3),(4,5),(6,7),(8,9)]

def _cl_binary_split(X_tr, y_tr, X_te, y_te, d0, d1, rng):
    """Balanced binary train/test split: label=1 if digit==d1."""
    tr_m = (y_tr == d0) | (y_tr == d1)
    te_m = (y_te == d0) | (y_te == d1)
    Xtr = X_tr[tr_m]; ytr = (y_tr[tr_m] == d1).astype(int)
    Xte = X_te[te_m]; yte = (y_te[te_m] == d1).astype(int)
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    n = min(len(pos), len(neg))
    idx = rng.permutation(np.concatenate([
        rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]))
    return Xtr[idx], ytr[idx], Xte, yte

def _dims_flops(dims):
    """Approximate FLOPs for one forward pass through an MLP (multiply-adds × 2)."""
    return sum(2 * dims[i] * dims[i+1] for i in range(len(dims)-1))

def run_continual_bench():
    import time as _time
    rng = np.random.default_rng(42)
    task_labels = [f'{a} vs {b}' for a, b in CL_TASKS]
    yield emit({'e':'cl_init','tasks':task_labels})

    # ── Load MNIST ─────────────────────────────────────────────
    yield emit({'e':'cl_loading'})
    try:
        base = './data/MNIST/raw'
        X_tr = _read_mnist_idx(f'{base}/train-images-idx3-ubyte.gz').reshape(-1,784).astype(np.float64)/255.0
        y_tr = _read_mnist_idx(f'{base}/train-labels-idx1-ubyte.gz').astype(int)
        X_te = _read_mnist_idx(f'{base}/t10k-images-idx3-ubyte.gz').reshape(-1,784).astype(np.float64)/255.0
        y_te = _read_mnist_idx(f'{base}/t10k-labels-idx1-ubyte.gz').astype(int)
        mu = X_tr.mean(0); sig = X_tr.std(0) + 1e-8
        X_tr = (X_tr - mu) / sig; X_te = (X_te - mu) / sig
    except FileNotFoundError:
        yield emit({'e':'cl_error','msg':'Run mnist_stress.py first to download MNIST'}); return
    except Exception as ex:
        yield emit({'e':'cl_error','msg':str(ex)}); return

    task_splits = [_cl_binary_split(X_tr, y_tr, X_te, y_te, d0, d1, rng)
                   for d0, d1 in CL_TASKS]
    yield emit({'e':'cl_loaded','n_tasks':5,
                'n_train':[len(s[0]) for s in task_splits],
                'n_test' :[len(s[2]) for s in task_splits]})

    Xr = np.vstack([task_splits[t][0][:500] for t in range(5)])
    dr = np.concatenate([np.full(min(500,len(task_splits[t][0])),t) for t in range(5)])

    # ── DAS: sequential leaf grafting ──────────────────────────
    forest = DASForest(784, CL_LEAF, num_leaves=5, seed=7)
    router_params = forest.router.W.size + forest.router.b.size

    t0 = _time.time()
    yield emit({'e':'cl_phase','phase':'das_router',
                'label':'Router: learning 5 task boundaries (Split-MNIST)'})
    for s in range(CL_ROUTER_STEPS):
        idx = rng.integers(0,len(Xr),CL_B)
        _, a = forest.router.train_step(Xr[idx], dr[idx], lr=CL_LR)
        if s % 80 == 0:
            yield emit({'e':'cl_router_step','step':s,'acc':round(a,4)})
    t_das_router = round(_time.time() - t0, 1)
    yield emit({'e':'cl_router_done'})

    das_matrix = [[None]*5 for _ in range(5)]
    t_das_leaves = []
    for t in range(5):
        d0, d1 = CL_TASKS[t]
        Xtr, ytr, Xte, yte = task_splits[t]
        yield emit({'e':'cl_phase','phase':'das_leaf','task':t,
                    'label':f'DAS grafts Leaf {t} — task {d0} vs {d1}'})
        for prev in range(t): forest.leaves[prev].frozen = True
        leaf = forest.leaves[t]; leaf.frozen = False
        t1 = _time.time()
        for s in range(CL_LEAF_STEPS):
            idx = rng.integers(0,len(Xtr),CL_B)
            leaf.backward(ce_grad(leaf.forward(Xtr[idx]),ytr[idx]), CL_LR)
            if s % 60 == 0:
                yield emit({'e':'cl_leaf_step','task':t,'step':s,
                            'acc':round(acc(leaf.forward(Xtr[:300]),ytr[:300]),4)})
        leaf.frozen = True
        t_das_leaves.append(round(_time.time()-t1, 1))
        for ev in range(t+1):
            Xe, ye = task_splits[ev][2], task_splits[ev][3]
            das_matrix[t][ev] = round(acc(forest.leaves[ev].forward(Xe),ye),4)
        yield emit({'e':'cl_das_eval','stage':t,'accs':das_matrix[t][:t+1]})

    das_leaf_params = n_params(forest.leaves[0])
    das_stored_params  = router_params + 5 * das_leaf_params
    das_active_params  = router_params + das_leaf_params   # only 1 leaf at inference
    das_infer_flops    = _dims_flops([forest.router.W.shape[0]] +
                                     [forest.router.W.shape[1]]) + _dims_flops(CL_LEAF)

    # ── Cross-domain contamination test ────────────────────────
    # Run leaf i on EVERY task's test set. The diagonal (own domain) should be
    # high; off-diagonal should collapse toward chance (~50% balanced binary).
    # This proves each leaf is a real specialist — and that the router (which
    # picks the diagonal) is doing essential work, not routing everything to one leaf.
    yield emit({'e':'cl_phase','phase':'contam',
                'label':'Cross-domain contamination test — each leaf on every domain'})
    contam = [[round(acc(forest.leaves[i].forward(task_splits[j][2]), task_splits[j][3]), 4)
               for j in range(5)] for i in range(5)]
    diag = [contam[i][i] for i in range(5)]
    off  = [contam[i][j] for i in range(5) for j in range(5) if i != j]
    yield emit({'e':'cl_contam','matrix':contam,
                'diag_mean':round(sum(diag)/len(diag), 4),
                'off_mean':round(sum(off)/len(off), 4)})

    # ── Fine-tuned MLP ─────────────────────────────────────────
    ft_mlp = FibonacciLeaf(CL_LEAF, seed=42)
    ft_matrix = [[None]*5 for _ in range(5)]
    t_ft_tasks = []
    for t in range(5):
        d0, d1 = CL_TASKS[t]
        Xtr, ytr, Xte, yte = task_splits[t]
        yield emit({'e':'cl_phase','phase':'finetune','task':t,
                    'label':f'Fine-tuned MLP overwrites on task {d0} vs {d1} (forgetting expected)'})
        t1 = _time.time()
        for s in range(CL_FT_STEPS):
            idx = rng.integers(0,len(Xtr),CL_B)
            ft_mlp.backward(ce_grad(ft_mlp.forward(Xtr[idx]),ytr[idx]), CL_LR)
            if s % 60 == 0:
                yield emit({'e':'cl_ft_step','task':t,'step':s,
                            'acc':round(acc(ft_mlp.forward(Xtr[:300]),ytr[:300]),4)})
        t_ft_tasks.append(round(_time.time()-t1, 1))
        for ev in range(t+1):
            Xe, ye = task_splits[ev][2], task_splits[ev][3]
            ft_matrix[t][ev] = round(acc(ft_mlp.forward(Xe),ye),4)
        yield emit({'e':'cl_ft_eval','stage':t,'accs':ft_matrix[t][:t+1]})

    ft_params_total = n_params(ft_mlp)
    ft_infer_flops  = _dims_flops(CL_LEAF)

    # ── EWC MLP: the real continual-learning baseline ──────────
    # Same architecture and init as the fine-tuned MLP, but after each task we
    # consolidate (store θ* + Fisher) and add the EWC penalty on later tasks.
    # EWC should forget LESS than naive fine-tuning, but — unlike DAS — it cannot
    # guarantee zero forgetting, and it trades plasticity (new-task accuracy) for it.
    ewc_mlp = FibonacciLeaf(CL_LEAF, seed=42)
    ewc_matrix = [[None]*5 for _ in range(5)]
    ewc_tasks = []          # consolidated {'fisher','star'} per finished task
    t_ewc_tasks = []
    for t in range(5):
        d0, d1 = CL_TASKS[t]
        Xtr, ytr, Xte, yte = task_splits[t]
        yield emit({'e':'cl_phase','phase':'ewc','task':t,
                    'label':f'EWC MLP on task {d0} vs {d1} — penalty protects old weights'})
        t1 = _time.time()
        for s in range(CL_FT_STEPS):
            idx = rng.integers(0,len(Xtr),CL_B)
            ewc_mlp.backward(ce_grad(ewc_mlp.forward(Xtr[idx]), ytr[idx]), CL_LR,
                             ewc_lambda=CL_EWC_LAMBDA, ewc_tasks=ewc_tasks)
            if s % 60 == 0:
                yield emit({'e':'cl_ewc_step','task':t,'step':s,
                            'acc':round(acc(ewc_mlp.forward(Xtr[:300]),ytr[:300]),4)})
        t_ewc_tasks.append(round(_time.time()-t1, 1))
        # Consolidate this task: anchor weights + Fisher importance
        ewc_tasks.append({'fisher': ewc_mlp.fisher_diagonal(Xtr, ytr, ce_grad, seed=t),
                          'star':   ewc_mlp.snapshot()})
        for ev in range(t+1):
            Xe, ye = task_splits[ev][2], task_splits[ev][3]
            ewc_matrix[t][ev] = round(acc(ewc_mlp.forward(Xe),ye),4)
        yield emit({'e':'cl_ewc_eval','stage':t,'accs':ewc_matrix[t][:t+1]})

    ewc_params_total = n_params(ewc_mlp)
    t_ewc_total = round(sum(t_ewc_tasks), 1)

    # ── PackNet: iterative pruning + per-task binary masks ──────
    # Same zero-forgetting guarantee as DAS (each task's weights get frozen
    # once claimed) but inside ONE fixed-size network instead of a new leaf
    # per task. Capacity is finite and gets carved up as tasks arrive, so
    # later tasks should get less of it — watch plasticity decline below.
    pkn = PackNetMLP(CL_LEAF, seed=42)
    pkn_matrix = [[None]*5 for _ in range(5)]
    pkn_free_counts = []
    t_pkn_tasks = []
    for t in range(5):
        d0, d1 = CL_TASKS[t]
        Xtr, ytr, Xte, yte = task_splits[t]
        yield emit({'e':'cl_phase','phase':'packnet','task':t,
                    'label':f'PackNet on task {d0} vs {d1} — claims its slice of fixed capacity'})
        t1 = _time.time()
        free_before = pkn.free_count()
        free_mask = [(own == -1) for own in pkn.owner]
        prng = np.random.default_rng(2000 + t)
        # 1) train on free weights only
        for s in range(CL_PKN_STEPS):
            idx = prng.integers(0, len(Xtr), CL_B)
            logits = pkn._forward(Xtr[idx])
            gW, gb = pkn._grads(ce_grad(logits, ytr[idx]))
            for i in range(len(pkn.W)):
                pkn.W[i] -= CL_LR * gW[i] * free_mask[i]
                pkn.b[i] -= CL_LR * gb[i]
            if s % 60 == 0:
                pkn.task_bias[t] = [bb.copy() for bb in pkn.b]   # temp bias for live eval
                yield emit({'e':'cl_pkn_step','task':t,'step':s,
                            'acc':round(acc(pkn.forward_task(Xtr[:300], t),ytr[:300]),4)})
        # 2) prune: claim the top |weight| slice of what's still free for task t
        remaining = 5 - t
        free_vals = np.concatenate([np.abs(pkn.W[i][free_mask[i]]) for i in range(len(pkn.W))])
        n_free = free_vals.size
        keep_count = int(round(n_free / max(remaining, 1)))
        thresh = (np.partition(free_vals, max(n_free-keep_count, 0))[max(n_free-keep_count, 0)]
                  if keep_count > 0 and n_free > 0 else np.inf)
        for i in range(len(pkn.W)):
            claim = free_mask[i] & (np.abs(pkn.W[i]) >= thresh)
            pkn.owner[i][claim] = t
        # 3) brief re-finetune restricted to task-t-owned weights, to recover
        #    accuracy lost from freezing the rest of the gradient signal away
        owned_mask = [(own == t) for own in pkn.owner]
        for s in range(CL_PKN_REFIT):
            idx = prng.integers(0, len(Xtr), CL_B)
            logits = pkn._forward(Xtr[idx])
            gW, gb = pkn._grads(ce_grad(logits, ytr[idx]))
            for i in range(len(pkn.W)):
                pkn.W[i] -= CL_LR * gW[i] * owned_mask[i]
                pkn.b[i] -= CL_LR * gb[i]
        pkn.task_bias[t] = [bb.copy() for bb in pkn.b]   # final bias snapshot for task t
        t_pkn_tasks.append(round(_time.time()-t1, 1))
        free_after = pkn.free_count()
        pkn_free_counts.append(free_after)
        for ev in range(t+1):
            Xe, ye = task_splits[ev][2], task_splits[ev][3]
            pkn_matrix[t][ev] = round(acc(pkn.forward_task(Xe, ev),ye),4)
        yield emit({'e':'cl_pkn_eval','stage':t,'accs':pkn_matrix[t][:t+1],
                    'free_before':int(free_before),'free_after':int(free_after)})

    pkn_params = n_params(pkn)
    t_pkn_total = round(sum(t_pkn_tasks), 1)

    # ── Multi-task MLP: upper bound ─────────────────────────────
    yield emit({'e':'cl_phase','phase':'multitask',
                'label':'Multi-task MLP: trained on all 10 digits at once (upper bound)'})
    mt_mlp = FibonacciLeaf(CL_LEAF_10, seed=7)
    t1 = _time.time()
    for s in range(CL_MT_STEPS):
        idx = rng.integers(0,len(X_tr),CL_B)
        mt_mlp.backward(ce_grad(mt_mlp.forward(X_tr[idx]),y_tr[idx]), CL_LR)
        if s % 100 == 0:
            a_all = acc(mt_mlp.forward(X_tr[:500]),y_tr[:500])
            yield emit({'e':'cl_mt_step','step':s,'acc':round(a_all,4)})
    t_mt = round(_time.time()-t1, 1)
    mt_params_total = n_params(mt_mlp)
    mt_infer_flops  = _dims_flops(CL_LEAF_10)

    mt_accs = []
    for t, (d0, d1) in enumerate(CL_TASKS):
        Xe, ye = task_splits[t][2], task_splits[t][3]
        logits = mt_mlp.forward(Xe)
        pred = (logits[:,d1] > logits[:,d0]).astype(int)
        mt_accs.append(round(float((pred==ye).mean()),4))
    yield emit({'e':'cl_mt_done','accs':mt_accs})

    # ── Derived metrics ─────────────────────────────────────────
    das_bwt = round(sum(das_matrix[4][i] - das_matrix[i][i] for i in range(4))/4, 4)
    ft_bwt  = round(sum(ft_matrix[4][i]  - ft_matrix[i][i]  for i in range(4))/4, 4)
    ewc_bwt = round(sum(ewc_matrix[4][i] - ewc_matrix[i][i] for i in range(4))/4, 4)
    pkn_bwt = round(sum(pkn_matrix[4][i] - pkn_matrix[i][i] for i in range(4))/4, 4)

    # Plasticity: accuracy when FIRST learning each task (diagonal)
    das_plasticity = [das_matrix[t][t] for t in range(5)]
    ft_plasticity  = [ft_matrix[t][t]  for t in range(5)]
    ewc_plasticity = [ewc_matrix[t][t] for t in range(5)]
    pkn_plasticity = [pkn_matrix[t][t] for t in range(5)]
    mt_plasticity  = mt_accs  # upper bound

    # Final accuracy: row 4 of each matrix
    das_final = [das_matrix[4][t] for t in range(5)]
    ft_final  = [ft_matrix[4][t]  for t in range(5)]
    ewc_final = [ewc_matrix[4][t] for t in range(5)]
    pkn_final = [pkn_matrix[4][t] for t in range(5)]

    # Stability = mean(final/first_learned) for tasks 0..3
    das_stability = round(sum(das_matrix[4][i]/max(das_matrix[i][i],1e-6) for i in range(4))/4, 4)
    ft_stability  = round(sum(ft_matrix[4][i] /max(ft_matrix[i][i], 1e-6) for i in range(4))/4, 4)
    ewc_stability = round(sum(ewc_matrix[4][i]/max(ewc_matrix[i][i],1e-6) for i in range(4))/4, 4)
    pkn_stability = round(sum(pkn_matrix[4][i]/max(pkn_matrix[i][i],1e-6) for i in range(4))/4, 4)

    # Forward transfer: average over tasks 1-4 of (acc_first_learned / acc_task0_at_same_stage)
    # Simplified: just emit raw numbers, compute in JS

    # Compute efficiency ratio: active_flops(DAS) / total_flops(FT)
    flops_ratio = round(das_infer_flops / max(ft_infer_flops, 1), 4)

    t_das_total = round(t_das_router + sum(t_das_leaves), 1)
    t_ft_total  = round(sum(t_ft_tasks), 1)

    yield emit({'e':'cl_done',
                'das_matrix':das_matrix,
                'ft_matrix':ft_matrix,
                'ewc_matrix':ewc_matrix,
                'pkn_matrix':pkn_matrix,
                'contam':contam,
                'mt_accs':mt_accs,
                'das_bwt':das_bwt,
                'ft_bwt':ft_bwt,
                'ewc_bwt':ewc_bwt,
                'pkn_bwt':pkn_bwt,
                'tasks':task_labels,
                # cost metrics
                'cost':{
                    'das_stored_params' : das_stored_params,
                    'das_active_params' : das_active_params,
                    'das_leaf_params'   : das_leaf_params,
                    'router_params'     : router_params,
                    'ft_params'         : ft_params_total,
                    'ewc_params'        : ewc_params_total,
                    'pkn_params'        : pkn_params,
                    'mt_params'         : mt_params_total,
                    'das_infer_flops'   : das_infer_flops,
                    'ft_infer_flops'    : ft_infer_flops,
                    'mt_infer_flops'    : mt_infer_flops,
                    'flops_ratio'       : flops_ratio,
                    't_das_router'      : t_das_router,
                    't_das_leaves'      : t_das_leaves,
                    't_das_total'       : t_das_total,
                    't_ft_tasks'        : t_ft_tasks,
                    't_ft_total'        : t_ft_total,
                    't_ewc_total'       : t_ewc_total,
                    't_pkn_total'       : t_pkn_total,
                    't_mt'              : t_mt,
                },
                # extra metrics
                'metrics':{
                    'das_plasticity'  : das_plasticity,
                    'ft_plasticity'   : ft_plasticity,
                    'ewc_plasticity'  : ewc_plasticity,
                    'pkn_plasticity'  : pkn_plasticity,
                    'mt_plasticity'   : mt_plasticity,
                    'das_final'       : das_final,
                    'ft_final'        : ft_final,
                    'ewc_final'       : ewc_final,
                    'pkn_final'       : pkn_final,
                    'das_stability'   : das_stability,
                    'ft_stability'    : ft_stability,
                    'ewc_stability'   : ewc_stability,
                    'pkn_stability'   : pkn_stability,
                    'pkn_free_counts' : pkn_free_counts,
                }})


# ── Phase 6: Permuted-MNIST — domain-incremental counterweight ──
# Split-MNIST is the regime where DAS's BWT≈0 looks most dramatic because EWC and
# fine-tuning genuinely struggle there (task-incremental, disjoint label pairs,
# heavy interference). Permuted-MNIST is the honest counter-example: every task is
# the SAME 10-class digit problem, just with a fixed pixel shuffle applied. This is
# domain-incremental, not task-incremental — and it is the well-known regime where
# EWC is expected to be much more competitive (Kirkpatrick et al. 2017 demoed EWC
# on exactly this benchmark). Including it keeps the benchmark suite honest: DAS
# still wins on BWT, but by a smaller margin, and that's reported plainly below.
PM_N_PERM      = 5
PM_LEAF_10     = [784, 128, 64, 10]   # one 10-class leaf per permutation
PM_LR          = 0.02; PM_B = 128
PM_N_TRAIN_SUB = 6000   # subset per task — keeps the web run ~70-90s
PM_N_TEST_SUB  = 1000
PM_ROUTER_STEPS = 800
PM_TASK_STEPS   = 800    # DAS / fine-tuned / EWC / PackNet steps per task
PM_PKN_REFIT    = 200
PM_MT_STEPS     = 1500
PM_EWC_LAMBDA   = 5000.0

def run_permuted_bench():
    import time as _time
    rng = np.random.default_rng(42)
    yield emit({'e':'pm_init','n_perm':PM_N_PERM})

    # ── Load MNIST + subset (full 10-class, NOT binary-split like Split-MNIST) ──
    yield emit({'e':'pm_loading'})
    try:
        base = './data/MNIST/raw'
        X_tr = _read_mnist_idx(f'{base}/train-images-idx3-ubyte.gz').reshape(-1,784).astype(np.float64)/255.0
        y_tr = _read_mnist_idx(f'{base}/train-labels-idx1-ubyte.gz').astype(int)
        X_te = _read_mnist_idx(f'{base}/t10k-images-idx3-ubyte.gz').reshape(-1,784).astype(np.float64)/255.0
        y_te = _read_mnist_idx(f'{base}/t10k-labels-idx1-ubyte.gz').astype(int)
        # NB: deliberately NOT z-score standardised here. Permuted-MNIST's router
        # signal comes from "is there ink at this (shuffled) pixel coordinate" —
        # raw [0,1] intensities preserve that; standardising per-original-position
        # before permuting doesn't change the underlying signal but raw pixels make
        # the magnitude-based PackNet pruning and the router both behave more
        # predictably, and it matches how this benchmark is usually reported.
    except FileNotFoundError:
        yield emit({'e':'pm_error','msg':'Run mnist_stress.py first to download MNIST'}); return
    except Exception as ex:
        yield emit({'e':'pm_error','msg':str(ex)}); return

    idx_tr = rng.choice(len(X_tr), PM_N_TRAIN_SUB, replace=False)
    idx_te = rng.choice(len(X_te), PM_N_TEST_SUB, replace=False)
    Xs_tr, ys_tr = X_tr[idx_tr], y_tr[idx_tr]
    Xs_te, ys_te = X_te[idx_te], y_te[idx_te]
    perms = [rng.permutation(784) for _ in range(PM_N_PERM)]
    yield emit({'e':'pm_loaded','n_train':PM_N_TRAIN_SUB,'n_test':PM_N_TEST_SUB,'n_perm':PM_N_PERM})

    def Xp(t, X): return X[:, perms[t]]   # apply permutation t to a batch

    # ── Router: N_PERM-way, distinguishing WHICH permutation produced a sample ──
    # A fixed permutation moves MNIST's mostly-zero border pixels to different
    # coordinates per task, so a linear router has a genuine, generalizing signal
    # to exploit here — it is not just memorising training rows.
    t0 = _time.time()
    yield emit({'e':'pm_phase','phase':'das_router',
                'label':f'Router: learning {PM_N_PERM} permutation identities'})
    Xr = np.vstack([Xp(t, Xs_tr) for t in range(PM_N_PERM)])
    dr = np.concatenate([np.full(len(Xs_tr), t) for t in range(PM_N_PERM)])
    router = StemRouter(784, PM_N_PERM, seed=7)
    for s in range(PM_ROUTER_STEPS):
        idx = rng.integers(0, len(Xr), PM_B)
        _, a = router.train_step(Xr[idx], dr[idx], lr=0.3)
        if s % 80 == 0:
            yield emit({'e':'pm_router_step','step':s,'acc':round(a,4)})
    t_pm_router = round(_time.time()-t0, 1)
    ridx, _ = router.route(Xr)
    router_acc = round(float((ridx==dr).mean()), 4)
    router_params = router.W.size + router.b.size
    yield emit({'e':'pm_router_done','acc':router_acc})

    # ── DAS: one 10-class leaf per permutation, frozen once trained ──────
    das_matrix = [[None]*PM_N_PERM for _ in range(PM_N_PERM)]
    das_leaves = []
    t_pm_das_leaves = []
    for t in range(PM_N_PERM):
        yield emit({'e':'pm_phase','phase':'das_leaf','task':t,
                    'label':f'DAS grafts Leaf {t} — permutation {t}'})
        leaf = FibonacciLeaf(PM_LEAF_10, seed=1+t)
        Xt = Xp(t, Xs_tr)
        t1 = _time.time()
        for s in range(PM_TASK_STEPS):
            idx = rng.integers(0, len(Xt), PM_B)
            leaf.backward(ce_grad(leaf.forward(Xt[idx]), ys_tr[idx]), PM_LR)
            if s % 80 == 0:
                yield emit({'e':'pm_leaf_step','task':t,'step':s,
                            'acc':round(acc(leaf.forward(Xt[:300]),ys_tr[:300]),4)})
        leaf.frozen = True
        das_leaves.append(leaf)
        t_pm_das_leaves.append(round(_time.time()-t1, 1))
        for ev in range(t+1):
            das_matrix[t][ev] = round(acc(das_leaves[ev].forward(Xp(ev, Xs_te)), ys_te), 4)
        yield emit({'e':'pm_das_eval','stage':t,'accs':das_matrix[t][:t+1]})

    das_leaf_params = n_params(das_leaves[0])
    das_stored_params = router_params + PM_N_PERM * das_leaf_params
    das_active_params = router_params + das_leaf_params
    das_infer_flops = _dims_flops([784, PM_N_PERM]) + _dims_flops(PM_LEAF_10)

    # ── Fine-tuned MLP: one shared net, naive sequential training ────────
    ft_mlp = FibonacciLeaf(PM_LEAF_10, seed=42)
    ft_matrix = [[None]*PM_N_PERM for _ in range(PM_N_PERM)]
    t_pm_ft_tasks = []
    for t in range(PM_N_PERM):
        yield emit({'e':'pm_phase','phase':'finetune','task':t,
                    'label':f'Fine-tuned MLP overwrites on permutation {t}'})
        Xt = Xp(t, Xs_tr)
        t1 = _time.time()
        for s in range(PM_TASK_STEPS):
            idx = rng.integers(0, len(Xt), PM_B)
            ft_mlp.backward(ce_grad(ft_mlp.forward(Xt[idx]), ys_tr[idx]), PM_LR)
            if s % 80 == 0:
                yield emit({'e':'pm_ft_step','task':t,'step':s,
                            'acc':round(acc(ft_mlp.forward(Xt[:300]),ys_tr[:300]),4)})
        t_pm_ft_tasks.append(round(_time.time()-t1, 1))
        for ev in range(t+1):
            ft_matrix[t][ev] = round(acc(ft_mlp.forward(Xp(ev, Xs_te)), ys_te), 4)
        yield emit({'e':'pm_ft_eval','stage':t,'accs':ft_matrix[t][:t+1]})

    ft_params_total = n_params(ft_mlp)
    ft_infer_flops = _dims_flops(PM_LEAF_10)
    t_pm_ft_total = round(sum(t_pm_ft_tasks), 1)

    # ── EWC MLP: the regime DAS's gap is expected to narrow against ──────
    ewc_mlp = FibonacciLeaf(PM_LEAF_10, seed=42)
    ewc_matrix = [[None]*PM_N_PERM for _ in range(PM_N_PERM)]
    ewc_tasks = []
    t_pm_ewc_tasks = []
    for t in range(PM_N_PERM):
        yield emit({'e':'pm_phase','phase':'ewc','task':t,
                    'label':f'EWC MLP on permutation {t} — penalty protects old weights'})
        Xt = Xp(t, Xs_tr)
        t1 = _time.time()
        for s in range(PM_TASK_STEPS):
            idx = rng.integers(0, len(Xt), PM_B)
            ewc_mlp.backward(ce_grad(ewc_mlp.forward(Xt[idx]), ys_tr[idx]), PM_LR,
                             ewc_lambda=PM_EWC_LAMBDA, ewc_tasks=ewc_tasks)
            if s % 80 == 0:
                yield emit({'e':'pm_ewc_step','task':t,'step':s,
                            'acc':round(acc(ewc_mlp.forward(Xt[:300]),ys_tr[:300]),4)})
        t_pm_ewc_tasks.append(round(_time.time()-t1, 1))
        ewc_tasks.append({'fisher': ewc_mlp.fisher_diagonal(Xt, ys_tr, ce_grad, seed=t),
                          'star':   ewc_mlp.snapshot()})
        for ev in range(t+1):
            ewc_matrix[t][ev] = round(acc(ewc_mlp.forward(Xp(ev, Xs_te)), ys_te), 4)
        yield emit({'e':'pm_ewc_eval','stage':t,'accs':ewc_matrix[t][:t+1]})

    ewc_params_total = n_params(ewc_mlp)
    t_pm_ewc_total = round(sum(t_pm_ewc_tasks), 1)

    # ── PackNet: same fixed-capacity pruning baseline as Split-MNIST ─────
    pkn = PackNetMLP(PM_LEAF_10, seed=42)
    pkn_matrix = [[None]*PM_N_PERM for _ in range(PM_N_PERM)]
    pkn_free_counts = []
    t_pm_pkn_tasks = []
    for t in range(PM_N_PERM):
        yield emit({'e':'pm_phase','phase':'packnet','task':t,
                    'label':f'PackNet on permutation {t} — claims its slice of fixed capacity'})
        Xt = Xp(t, Xs_tr)
        t1 = _time.time()
        free_before = pkn.free_count()
        free_mask = [(own == -1) for own in pkn.owner]
        prng = np.random.default_rng(3000 + t)
        for s in range(PM_TASK_STEPS):
            idx = prng.integers(0, len(Xt), PM_B)
            logits = pkn._forward(Xt[idx])
            gW, gb = pkn._grads(ce_grad(logits, ys_tr[idx]))
            for i in range(len(pkn.W)):
                pkn.W[i] -= PM_LR * gW[i] * free_mask[i]
                pkn.b[i] -= PM_LR * gb[i]
            if s % 80 == 0:
                pkn.task_bias[t] = [bb.copy() for bb in pkn.b]
                yield emit({'e':'pm_pkn_step','task':t,'step':s,
                            'acc':round(acc(pkn.forward_task(Xt[:300], t),ys_tr[:300]),4)})
        remaining = PM_N_PERM - t
        free_vals = np.concatenate([np.abs(pkn.W[i][free_mask[i]]) for i in range(len(pkn.W))])
        n_free = free_vals.size
        keep_count = int(round(n_free / max(remaining, 1)))
        thresh = (np.partition(free_vals, max(n_free-keep_count, 0))[max(n_free-keep_count, 0)]
                  if keep_count > 0 and n_free > 0 else np.inf)
        for i in range(len(pkn.W)):
            claim = free_mask[i] & (np.abs(pkn.W[i]) >= thresh)
            pkn.owner[i][claim] = t
        owned_mask = [(own == t) for own in pkn.owner]
        for s in range(PM_PKN_REFIT):
            idx = prng.integers(0, len(Xt), PM_B)
            logits = pkn._forward(Xt[idx])
            gW, gb = pkn._grads(ce_grad(logits, ys_tr[idx]))
            for i in range(len(pkn.W)):
                pkn.W[i] -= PM_LR * gW[i] * owned_mask[i]
                pkn.b[i] -= PM_LR * gb[i]
        pkn.task_bias[t] = [bb.copy() for bb in pkn.b]
        t_pm_pkn_tasks.append(round(_time.time()-t1, 1))
        free_after = pkn.free_count()
        pkn_free_counts.append(free_after)
        for ev in range(t+1):
            pkn_matrix[t][ev] = round(acc(pkn.forward_task(Xp(ev, Xs_te), ev), ys_te), 4)
        yield emit({'e':'pm_pkn_eval','stage':t,'accs':pkn_matrix[t][:t+1],
                    'free_before':int(free_before),'free_after':int(free_after)})

    pkn_params = n_params(pkn)
    t_pm_pkn_total = round(sum(t_pm_pkn_tasks), 1)

    # ── Multi-task MLP: trained on all permutations mixed — upper bound ──
    yield emit({'e':'pm_phase','phase':'multitask',
                'label':'Multi-task MLP: trained on all permutations at once (upper bound)'})
    mt_mlp = FibonacciLeaf(PM_LEAF_10, seed=7)
    Xall = np.vstack([Xp(t, Xs_tr) for t in range(PM_N_PERM)])
    yall = np.concatenate([ys_tr for _ in range(PM_N_PERM)])
    t1 = _time.time()
    for s in range(PM_MT_STEPS):
        idx = rng.integers(0, len(Xall), PM_B)
        mt_mlp.backward(ce_grad(mt_mlp.forward(Xall[idx]), yall[idx]), PM_LR)
        if s % 150 == 0:
            yield emit({'e':'pm_mt_step','step':s,
                        'acc':round(acc(mt_mlp.forward(Xall[:500]),yall[:500]),4)})
    t_pm_mt = round(_time.time()-t1, 1)
    mt_params_total = n_params(mt_mlp)
    mt_infer_flops = _dims_flops(PM_LEAF_10)
    mt_accs = [round(acc(mt_mlp.forward(Xp(t, Xs_te)), ys_te), 4) for t in range(PM_N_PERM)]
    yield emit({'e':'pm_mt_done','accs':mt_accs})

    # ── Derived metrics — same formulas as Split-MNIST, NxN this time ────
    N = PM_N_PERM
    das_bwt = round(sum(das_matrix[N-1][i] - das_matrix[i][i] for i in range(N-1))/(N-1), 4)
    ft_bwt  = round(sum(ft_matrix[N-1][i]  - ft_matrix[i][i]  for i in range(N-1))/(N-1), 4)
    ewc_bwt = round(sum(ewc_matrix[N-1][i] - ewc_matrix[i][i] for i in range(N-1))/(N-1), 4)
    pkn_bwt = round(sum(pkn_matrix[N-1][i] - pkn_matrix[i][i] for i in range(N-1))/(N-1), 4)

    das_plasticity = [das_matrix[t][t] for t in range(N)]
    ft_plasticity  = [ft_matrix[t][t]  for t in range(N)]
    ewc_plasticity = [ewc_matrix[t][t] for t in range(N)]
    pkn_plasticity = [pkn_matrix[t][t] for t in range(N)]
    mt_plasticity  = mt_accs

    das_final = [das_matrix[N-1][t] for t in range(N)]
    ft_final  = [ft_matrix[N-1][t]  for t in range(N)]
    ewc_final = [ewc_matrix[N-1][t] for t in range(N)]
    pkn_final = [pkn_matrix[N-1][t] for t in range(N)]

    das_stability = round(sum(das_matrix[N-1][i]/max(das_matrix[i][i],1e-6) for i in range(N-1))/(N-1), 4)
    ft_stability  = round(sum(ft_matrix[N-1][i] /max(ft_matrix[i][i], 1e-6) for i in range(N-1))/(N-1), 4)
    ewc_stability = round(sum(ewc_matrix[N-1][i]/max(ewc_matrix[i][i],1e-6) for i in range(N-1))/(N-1), 4)
    pkn_stability = round(sum(pkn_matrix[N-1][i]/max(pkn_matrix[i][i],1e-6) for i in range(N-1))/(N-1), 4)

    flops_ratio = round(das_infer_flops / max(ft_infer_flops, 1), 4)
    t_pm_das_total = round(t_pm_router + sum(t_pm_das_leaves), 1)

    yield emit({'e':'pm_done',
                'das_matrix':das_matrix, 'ft_matrix':ft_matrix,
                'ewc_matrix':ewc_matrix, 'pkn_matrix':pkn_matrix,
                'mt_accs':mt_accs,
                'das_bwt':das_bwt, 'ft_bwt':ft_bwt, 'ewc_bwt':ewc_bwt, 'pkn_bwt':pkn_bwt,
                'router_acc':router_acc,
                'cost':{
                    'das_stored_params' : das_stored_params,
                    'das_active_params' : das_active_params,
                    'das_leaf_params'   : das_leaf_params,
                    'router_params'     : router_params,
                    'ft_params'         : ft_params_total,
                    'ewc_params'        : ewc_params_total,
                    'pkn_params'        : pkn_params,
                    'mt_params'         : mt_params_total,
                    'das_infer_flops'   : das_infer_flops,
                    'ft_infer_flops'    : ft_infer_flops,
                    'mt_infer_flops'    : mt_infer_flops,
                    'flops_ratio'       : flops_ratio,
                    't_das_router'      : t_pm_router,
                    't_das_leaves'      : t_pm_das_leaves,
                    't_das_total'       : t_pm_das_total,
                    't_ft_tasks'        : t_pm_ft_tasks,
                    't_ft_total'        : t_pm_ft_total,
                    't_ewc_total'       : t_pm_ewc_total,
                    't_pkn_total'       : t_pm_pkn_total,
                    't_mt'              : t_pm_mt,
                },
                'metrics':{
                    'das_plasticity'  : das_plasticity,
                    'ft_plasticity'   : ft_plasticity,
                    'ewc_plasticity'  : ewc_plasticity,
                    'pkn_plasticity'  : pkn_plasticity,
                    'mt_plasticity'   : mt_plasticity,
                    'das_final'       : das_final,
                    'ft_final'        : ft_final,
                    'ewc_final'       : ewc_final,
                    'pkn_final'       : pkn_final,
                    'das_stability'   : das_stability,
                    'ft_stability'    : ft_stability,
                    'ewc_stability'   : ewc_stability,
                    'pkn_stability'   : pkn_stability,
                    'pkn_free_counts' : pkn_free_counts,
                }})


@app.route('/')
def index(): return render_template('index.html')

@app.route('/stress')
def stress(): return render_template('stress.html')

@app.route('/train')
def train():
    return Response(run_training(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.route('/benchmark')
def benchmark():
    return Response(run_benchmark(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.route('/stress-stream')
def stress_stream():
    return Response(run_stress(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.route('/real')
def real_bench(): return render_template('real_bench.html')

@app.route('/real-stream')
def real_stream():
    return Response(run_real_bench(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.route('/continual')
def continual(): return render_template('continual_bench.html')

@app.route('/continual-stream')
def continual_stream():
    return Response(run_continual_bench(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.route('/permuted')
def permuted(): return render_template('permuted_bench.html')

@app.route('/permuted-stream')
def permuted_stream():
    return Response(run_permuted_bench(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

if __name__=='__main__':
    print("\n  → Forest + Benchmark: http://localhost:5050")
    print("  → MNIST Stress Test:  http://localhost:5050/stress\n")
    app.run(debug=False, port=5050, threaded=True)
