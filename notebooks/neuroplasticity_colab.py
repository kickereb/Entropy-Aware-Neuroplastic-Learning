#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NEUROPLASTICITY GROWING NEURAL NETWORK — Google Colab / PyTorch            ║
# ║                                                                              ║
# ║  Uses real CIFAR-10 via torchvision. GPU optional (runs in ~3 min on T4).  ║
# ║                                                                              ║
# ║  TO RUN IN COLAB:                                                            ║
# ║    1. Upload this file, or paste sections into cells                        ║
# ║    2. Runtime → Run all                                                      ║
# ║    3. Results figure saved to neuroplasticity_cifar10_results.png           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# %% [markdown]
# # Neuroplasticity-Inspired Growing Neural Network  ·  Real CIFAR-10
#
# A network that grows new neurons when it struggles, guided by four
# information-theoretic signals — simulating biological neuroplasticity.

# %% Cell 1 — Install / imports
# !pip install torch torchvision scipy matplotlib --quiet

import math, time, warnings, copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {DEVICE}")
print(f"PyTorch : {torch.__version__}")


# %% Cell 2 — CIFAR-10 dataset
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
test_set  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True,  num_workers=2)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=512, shuffle=False, num_workers=2)

# Flatten CIFAR-10 images: 3×32×32 = 3072 dims
# For the MLP experiment we use the raw flattened pixels.
# (For a CNN backbone, swap in features from a frozen ResNet-18 — see comments below.)
INPUT_DIM = 3 * 32 * 32   # 3072
N_CLASSES = 10

print(f"Train: {len(train_set):,} samples  |  Test: {len(test_set):,} samples")
print(f"Input dim: {INPUT_DIM}  |  Classes: {N_CLASSES}")


# %% Cell 3 — Optional CNN feature extractor (comment out to use raw pixels)
# Using a frozen ResNet-18 backbone produces 512-dim features that are
# much more linearly separable than raw pixels — this makes the growth
# dynamics cleaner. Uncomment the block below to use it.
#
# import torchvision.models as models
#
# backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# backbone.fc = nn.Identity()   # strip the classifier head
# backbone = backbone.to(DEVICE).eval()
#
# @torch.no_grad()
# def extract_features(loader):
#     feats, labels = [], []
#     for x, y in loader:
#         feats.append(backbone(x.to(DEVICE)).cpu())
#         labels.append(y)
#     return torch.cat(feats), torch.cat(labels)
#
# print("Extracting ResNet-18 features ...")
# X_tr_feat, y_tr = extract_features(train_loader)
# X_te_feat, y_te = extract_features(test_loader)
# INPUT_DIM = 512
# print(f"Features: {X_tr_feat.shape}  |  Labels: {y_tr.shape}")
#
# # Move to device for training
# X_tr = X_tr_feat.to(DEVICE); y_tr = y_tr.to(DEVICE)
# X_te = X_te_feat.to(DEVICE); y_te = y_te.to(DEVICE)


# %% Cell 4 — Growing MLP (PyTorch)
class GrowingMLP(nn.Module):
    """
    Dynamically growing fully-connected network.
    grow_width() and grow_depth() return new instances with expanded weights.
    """

    def __init__(self, sizes: list):
        super().__init__()
        self.sizes   = list(sizes)
        self.linears = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1])
            for i in range(len(sizes) - 1)
        ])
        self._init_weights()

    def _init_weights(self):
        for L in self.linears:
            nn.init.kaiming_normal_(L.weight, nonlinearity="relu")
            nn.init.zeros_(L.bias)

    def forward(self, x):
        for i, L in enumerate(self.linears):
            x = L(x)
            if i < len(self.linears) - 1:
                x = torch.relu(x)
        return x  # raw logits

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return f"GrowingMLP(sizes={self.sizes}, params={self.n_params():,})"

    @torch.no_grad()
    def represent(self, X: torch.Tensor, batch: int = 512) -> np.ndarray:
        """Penultimate-layer activations as numpy float32."""
        self.eval()
        reps = []
        for s in range(0, len(X), batch):
            a = X[s:s + batch]
            for L in self.linears[:-1]:
                a = torch.relu(L(a))
            reps.append(a.cpu().float().numpy())
        return np.vstack(reps)


# %% Cell 5 — Net2Net growth operators
def grow_width(model: GrowingMLP, delta: int = 16, lr: float = 1.5e-3,
               old_opt=None) -> tuple:
    """
    Add `delta` neurons to every hidden layer.
    Returns (new_model, new_optimiser).
    """
    old    = model.sizes
    new_sz = [old[0]] + [s + delta for s in old[1:-1]] + [old[-1]]
    new_m  = GrowingMLP(new_sz).to(DEVICE)

    with torch.no_grad():
        for i, (src, dst) in enumerate(zip(model.linears, new_m.linears)):
            oi, oo = old[i + 1], old[i]   # out_features, in_features
            # Existing weights
            dst.weight[:oi, :oo] = src.weight.clone()
            dst.bias[:oi]        = src.bias.clone()
            # New output neurons (10% scale He)
            if new_sz[i + 1] > oi:
                ex  = new_sz[i + 1] - oi
                std = math.sqrt(2.0 / oo) * 0.1
                nn.init.normal_(dst.weight[oi:, :oo], std=std)
            # New input connections
            if i > 0 and new_sz[i] > oo:
                ei = new_sz[i] - oo
                dst.weight[:oi, oo:].fill_(0.0)
                dst.weight[:oi, oo:] += torch.randn(oi, ei, device=DEVICE) * 0.001
                if new_sz[i + 1] > oi:
                    dst.weight[oi:, oo:] = torch.randn(new_sz[i+1]-oi, ei, device=DEVICE) * 0.001

    new_opt = optim.Adam(new_m.parameters(), lr=lr)
    return new_m, new_opt


def grow_depth(model: GrowingMLP, lr: float = 1.5e-3) -> tuple:
    """
    Insert a new hidden layer before the output.
    Returns (new_model, new_optimiser).
    """
    old    = model.sizes
    new_h  = max(8, old[-2] // 2)
    new_sz = old[:-1] + [new_h, old[-1]]
    new_m  = GrowingMLP(new_sz).to(DEVICE)

    with torch.no_grad():
        for i in range(len(model.linears) - 1):
            new_m.linears[i].weight.copy_(model.linears[i].weight)
            new_m.linears[i].bias.copy_(model.linears[i].bias)
        # Near-identity new layer
        nd = new_m.linears[-2]
        mn = min(nd.weight.shape)
        nd.weight.zero_()
        nd.weight[:mn, :mn] = torch.eye(mn, device=DEVICE) * 0.1
        nd.bias.zero_()
        # Output layer
        new_m.linears[-1].bias.copy_(model.linears[-1].bias)
        nn.init.normal_(new_m.linears[-1].weight, std=0.01)

    new_opt = optim.Adam(new_m.parameters(), lr=lr)
    return new_m, new_opt


# %% Cell 6 — IT metrics (PyTorch-compatible)
def effective_rank(W: np.ndarray) -> float:
    _, sv, _ = np.linalg.svd(W.astype(np.float64), full_matrices=False)
    sv = sv[sv > 1e-9]
    if len(sv) == 0: return 1.0
    p = sv / sv.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))

def mean_eff_rank(model: GrowingMLP) -> float:
    return float(np.mean([
        effective_rank(L.weight.detach().cpu().numpy())
        for L in model.linears
    ]))

def fisher_trace(model: GrowingMLP, Xb: torch.Tensor, yb: torch.Tensor) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    criterion(model(Xb), yb).backward()
    tr = sum(
        float((p.grad ** 2).sum())
        for p in model.parameters() if p.grad is not None
    )
    model.zero_grad()
    return tr

def mutual_info_ib(reps: np.ndarray, y: np.ndarray, n_bins: int = 28) -> float:
    rc = (reps - reps.mean(0)).astype(np.float64)
    try:
        _, _, Vt = np.linalg.svd(rc, full_matrices=False)
        t = (rc @ Vt[0]).astype(np.float32)
    except Exception:
        t = rc[:, 0].astype(np.float32)
    lo, hi = float(t.min()) - 1e-6, float(t.max()) + 1e-6
    edges  = np.linspace(lo, hi, n_bins + 1)
    T_bin  = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    pT = np.bincount(T_bin, minlength=n_bins).astype(float) / len(y)
    HT = -np.sum(pT * np.log(pT + 1e-12))
    HTY = 0.0
    for c in range(N_CLASSES):
        mask = (y == c)
        if not mask.any(): continue
        pTc  = np.bincount(T_bin[mask], minlength=n_bins).astype(float)
        pTc /= pTc.sum() + 1e-12
        HTY += mask.mean() * (-np.sum(pTc * np.log(pTc + 1e-12)))
    return float(max(0.0, HT - HTY))

def twonn_id(X: np.ndarray, n_sub: int = 200) -> float:
    if X.shape[0] > n_sub:
        X = X[np.random.choice(X.shape[0], n_sub, replace=False)]
    X64 = X.astype(np.float64)
    D   = cdist(X64, X64); np.fill_diagonal(D, np.inf)
    ds  = np.sort(D, axis=1)
    mu  = (ds[:, 1] + 1e-12) / (ds[:, 0] + 1e-12); mu = mu[mu > 1.0]
    if len(mu) < 5: return 1.0
    return float(np.clip(1.0 / np.mean(np.log(mu)), 1.0, X64.shape[1]))


# %% Cell 7 — Neuroplasticity Controller
class NeuroplasticityController:
    def __init__(self, acc_thresh=0.60, delta_acc=0.006, cooldown=7):
        self.acc_thresh    = acc_thresh
        self.delta_acc     = delta_acc
        self.cooldown      = cooldown
        self._prev_acc     = 0.0
        self._last_growth  = -9999
        self.growth_accs   = []
        self.growth_kinds  = []
        self.width_count   = 0

    def should_grow(self, epoch, train_acc):
        low    = train_acc < self.acc_thresh
        stall  = abs(train_acc - self._prev_acc) < self.delta_acc
        cooled = (epoch - self._last_growth) >= self.cooldown
        self._prev_acc = train_acc
        return low and stall and cooled

    def prefer_depth(self):
        if self.width_count < 3 or len(self.growth_accs) < 3: return False
        gains = [self.growth_accs[i] - self.growth_accs[i-1]
                 for i in range(1, len(self.growth_accs))]
        return len(gains) >= 2 and all(g < 0.015 for g in gains[-2:])

    def record(self, kind, epoch, acc):
        self.growth_accs.append(acc); self.growth_kinds.append(kind)
        self._last_growth = epoch
        if kind == "width": self.width_count += 1


# %% Cell 8 — Training helpers
def train_epoch(model, loader, optimiser, criterion, flatten=True):
    model.train()
    tl, tc, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if flatten: x = x.view(x.size(0), -1)
        optimiser.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()
        tl += float(loss) * len(y); tc += int((model(x).argmax(1) == y).sum()); n += len(y)
    return tl / n, tc / n

@torch.no_grad()
def evaluate(model, loader, criterion, flatten=True):
    model.eval()
    tl, tc, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if flatten: x = x.view(x.size(0), -1)
        logits = model(x)
        tl += float(criterion(logits, y)) * len(y)
        tc += int((logits.argmax(1) == y).sum()); n += len(y)
    return tl / n, tc / n


# %% Cell 9 — Fixed analysis subsets for IT metrics
@torch.no_grad()
def get_subset(loader, n=400, flatten=True):
    Xs, ys = [], []
    for x, y in loader:
        Xs.append(x); ys.append(y)
        if sum(len(b) for b in Xs) >= n: break
    X = torch.cat(Xs)[:n]; y_np = torch.cat(ys)[:n].numpy()
    if flatten: X = X.view(X.size(0), -1)
    return X.to(DEVICE), y_np

X_it, y_it_np = get_subset(test_loader,  n=400)
X_fi, y_fi    = get_subset(train_loader, n=256)
y_fi          = y_fi.to(DEVICE) if hasattr(y_fi, "to") else torch.tensor(y_fi).to(DEVICE)

print(f"IT subsets ready: {X_it.shape} for ID/MI, {X_fi.shape} for Fisher")

print("\nComputing Data ID ...")
_sample = X_it.cpu().float().numpy()
DATA_ID = twonn_id(_sample, n_sub=200)
print(f"DATA_ID = {DATA_ID:.2f}")


# %% Cell 10 — Main training loop
# Config
INIT_SIZES   = [INPUT_DIM, 4, 4, N_CLASSES]
MAX_EPOCHS   = 80
WIDTH_DELTA  = 16
LR_INIT      = 1e-3
LR_POST      = 7e-4

model     = GrowingMLP(INIT_SIZES).to(DEVICE)
optimiser = optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
ctrl      = NeuroplasticityController(acc_thresh=0.50, delta_acc=0.005, cooldown=6)

print(f"\n{repr(model)}\n")
print("─" * 100)
print(f"{'Ep':>3}  {'TrLoss':>8} {'TrAcc':>7}  {'TeLoss':>8} {'TeAcc':>7}  "
      f"{'EffRank':>8} {'FIM(log)':>9} {'I(T;Y)':>7} {'RepID':>6}   Event")
print("─" * 100)

H = {k: [] for k in ["ep","tr_loss","tr_acc","te_loss","te_acc",
                      "n_params","sizes","eff_rank","fisher","mi","rep_id"]}
GROWTHS = []
t0 = time.time()

for ep in range(1, MAX_EPOCHS + 1):
    tr_loss, tr_acc = train_epoch(model, train_loader, optimiser, criterion)
    te_loss, te_acc = evaluate(model, test_loader, criterion)

    er   = mean_eff_rank(model)
    fi   = fisher_trace(model, X_fi, y_fi)
    reps = model.represent(X_it)
    mi   = mutual_info_ib(reps, y_it_np)
    rid  = twonn_id(reps, n_sub=200)

    H["ep"].append(ep);          H["tr_loss"].append(tr_loss)
    H["tr_acc"].append(tr_acc);  H["te_loss"].append(te_loss)
    H["te_acc"].append(te_acc);  H["n_params"].append(model.n_params())
    H["sizes"].append(model.sizes.copy())
    H["eff_rank"].append(er);    H["fisher"].append(float(np.log10(fi + 1e-20)))
    H["mi"].append(mi);          H["rep_id"].append(rid)

    ev = ""
    if ctrl.should_grow(ep, tr_acc):
        old_r = repr(model)
        if ctrl.prefer_depth():
            kind, model, optimiser = "depth", *grow_depth(model, lr=LR_POST)
            tag = "↕ DEPTH"
        else:
            kind, model, optimiser = "width", *grow_width(model, delta=WIDTH_DELTA, lr=LR_POST)
            tag = "↔ WIDTH"
        ctrl.record(kind, ep, tr_acc)
        GROWTHS.append((ep, kind, old_r, repr(model)))
        ev = f"  [{tag}] → {repr(model)}"

    print(f"{ep:3d}  {tr_loss:8.4f} {tr_acc*100:6.2f}%  "
          f"{te_loss:8.4f} {te_acc*100:6.2f}%  "
          f"{er:8.3f} {np.log10(fi+1e-20):9.3f} {mi:7.3f} {rid:6.2f}  {ev}")

print("─" * 100)
print(f"\nDone in {time.time()-t0:.1f}s  |  Growth events: {[(e,k) for e,k,_,_ in GROWTHS]}")
print(f"Final model: {repr(model)}")


# %% Cell 11 — Results figure
# Build in 3 strips to keep memory bounded, stitch with Pillow
import os, tempfile
from PIL import Image

BG, PBG = "#0d0d1c", "#131330"
C = {"train":"#4fc3f7","test":"#ef9a9a","width":"#69f0ae","depth":"#ce93d8",
     "data":"#90caf9","param":"#80deea","rank":"#ffcc80","fisher":"#ff8a65",
     "mi":"#b39ddb","id":"#4dd0e1"}
eps = H["ep"]

def sty(ax):
    ax.set_facecolor(PBG); ax.tick_params(colors="#ccd", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#252545")
    return ax

def vl(ax):
    yl = ax.get_ylim()
    for ep_g, kind, *_ in GROWTHS:
        col = C["width"] if kind=="width" else C["depth"]
        ax.axvline(ep_g, color=col, lw=1.4, ls="--", alpha=0.82, zorder=5)
        ax.text(ep_g, yl[0]+(yl[1]-yl[0])*0.94, "W↔" if kind=="width" else "D↕",
                color=col, ha="center", va="top", fontsize=7.5, fontweight="bold",
                transform=ax.get_xaxis_transform())

TC = dict(color="white", fontsize=10, fontweight="bold")
strip_paths = []
tmpdir = tempfile.mkdtemp()

# Strip 1
fig, axes = plt.subplots(1, 3, figsize=(21, 5.2), facecolor=BG)
fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97, top=0.88, bottom=0.13)
ax = sty(axes[0])
ax.plot(eps,[a*100 for a in H["tr_acc"]],C["train"],lw=2,label="Train")
ax.plot(eps,[a*100 for a in H["te_acc"]],C["test"], lw=2,ls="--",label="Test")
ax.axhline(ctrl.acc_thresh*100,color="#ffeb3b",lw=1.2,ls=":",alpha=0.75,label="Growth threshold")
vl(ax); ax.set_title("Accuracy",**TC); ax.set_xlabel("Epoch",color="#aac"); ax.set_ylabel("Acc (%)",color="#aac")
ax.legend(fontsize=8, facecolor=PBG, labelcolor="white", framealpha=0.65)
ax = sty(axes[1])
ax.plot(eps,H["tr_loss"],C["train"],lw=2,label="Train")
ax.plot(eps,H["te_loss"],C["test"], lw=2,ls="--",label="Test")
vl(ax); ax.set_title("Cross-Entropy Loss",**TC); ax.set_xlabel("Epoch",color="#aac"); ax.set_ylabel("Loss",color="#aac")
ax.legend(fontsize=8, facecolor=PBG, labelcolor="white", framealpha=0.65)
ax = sty(axes[2])
ax.step(eps,H["n_params"],C["param"],lw=2.3,where="post")
ax.fill_between(eps,0,H["n_params"],step="post",color=C["param"],alpha=0.14)
vl(ax); ax.set_title("Parameter Count",**TC); ax.set_xlabel("Epoch",color="#aac"); ax.set_ylabel("Parameters",color="#aac")
fig.suptitle("Neuroplasticity Growing Neural Network  ·  Real CIFAR-10",
             fontsize=11, fontweight="bold", color="white", y=0.99)
p1 = os.path.join(tmpdir, "s1.png")
plt.savefig(p1, dpi=130, bbox_inches="tight", facecolor=BG); plt.close()
strip_paths.append(p1)

# Strip 2
fig, axes = plt.subplots(1, 3, figsize=(21, 5.2), facecolor=BG)
fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97, top=0.85, bottom=0.13)
ax = sty(axes[0]); ax.plot(eps,H["eff_rank"],C["rank"],lw=2,marker="o",ms=3); vl(ax)
ax.set_title("Mean Effective Rank\nexp(H(σ̃))",**TC); ax.set_xlabel("Epoch",color="#aac"); ax.set_ylabel("Eff. Rank",color="#aac")
ax = sty(axes[1]); ax.plot(eps,H["fisher"],C["fisher"],lw=2,marker="^",ms=3); vl(ax)
ax.set_title("Fisher Information Trace (log₁₀)\nE[(∂L/∂θ)²]",**TC); ax.set_xlabel("Epoch",color="#aac"); ax.set_ylabel("log₁₀ FIM",color="#aac")
ax = sty(axes[2]); ax.plot(eps,H["mi"],C["mi"],lw=2,marker="s",ms=3); vl(ax)
ax.set_title("Mutual Info I(T;Y) [nats]\nInformation Bottleneck",**TC); ax.set_xlabel("Epoch",color="#aac"); ax.set_ylabel("MI (nats)",color="#aac")
p2 = os.path.join(tmpdir, "s2.png")
plt.savefig(p2, dpi=130, bbox_inches="tight", facecolor=BG); plt.close()
strip_paths.append(p2)

# Strip 3
fig, axes = plt.subplots(1, 2, figsize=(21, 5.8), facecolor=BG, gridspec_kw={"width_ratios":[2,1]})
fig.subplots_adjust(wspace=0.32, left=0.06, right=0.97, top=0.83, bottom=0.13)
ax = sty(axes[0]); rid = H["rep_id"]; dl = [DATA_ID]*len(eps)
ax.fill_between(eps,rid,dl,where=[r<d for r,d in zip(rid,dl)],color="#f44336",alpha=0.18,label="Complexity gap")
ax.fill_between(eps,rid,dl,where=[r>=d for r,d in zip(rid,dl)],color="#4caf50",alpha=0.22,label="Model ≥ data")
ax.plot(eps,rid,C["id"],lw=2.8,marker="o",ms=4.5,label="Rep ID")
ax.axhline(DATA_ID,color=C["data"],lw=2,ls="--",label=f"Data ID={DATA_ID:.1f}")
for ep_g, kind, *_ in GROWTHS:
    col=C["width"] if kind=="width" else C["depth"]
    ax.axvline(ep_g,color=col,lw=1.5,ls="--",alpha=0.85,zorder=5)
    ax.text(ep_g,1.,"W↔" if kind=="width" else "D↕",color=col,ha="center",va="top",
            fontsize=8,fontweight="bold",transform=ax.get_xaxis_transform())
ax.set_title("🧠 NEUROPLASTICITY SIGNAL — Complexity Gap Closure\nRep ID vs. Data ID",color="white",fontsize=10,fontweight="bold")
ax.set_xlabel("Epoch",color="#aac"); ax.set_ylabel("Intrinsic Dimensionality",color="#aac")
ax.legend(fontsize=8,facecolor=PBG,labelcolor="white",framealpha=0.65,loc="lower right")
ax = sty(axes[1])
sizes_per_ep=[]; cur=list(INIT_SIZES); gi=0
for ep in eps:
    sizes_per_ep.append(cur[1:-1])
    if gi<len(GROWTHS) and GROWTHS[gi][0]==ep:
        k=GROWTHS[gi][1]
        cur=([cur[0]]+[s+WIDTH_DELTA for s in cur[1:-1]]+[cur[-1]] if k=="width" else cur[:-1]+[max(8,cur[-2]//2),cur[-1]]); gi+=1
max_l=max(len(h) for h in sizes_per_ep); pal=plt.cm.plasma(np.linspace(0.15,0.92,max_l)); bot=np.zeros(len(eps))
for l in range(max_l):
    v=np.array([h[l] if l<len(h) else 0 for h in sizes_per_ep],float)
    ax.bar(eps,v,bottom=bot,color=pal[l],label=f"H{l+1}",width=0.9); bot+=v
for ep_g,kind,*_ in GROWTHS: ax.axvline(ep_g,color=C["width"] if kind=="width" else C["depth"],lw=1.4,ls="--",alpha=0.85)
ax.set_title("Architecture Evolution",color="white",fontsize=10,fontweight="bold")
ax.set_xlabel("Epoch",color="#aac"); ax.set_ylabel("Hidden neurons",color="#aac")
ax.legend(fontsize=7,facecolor=PBG,labelcolor="white",ncol=2,framealpha=0.65)
n_w=sum(1 for _,k,*_ in GROWTHS if k=="width"); n_d=sum(1 for _,k,*_ in GROWTHS if k=="depth")
fig.legend(handles=[
    Line2D([0],[0],color=C["width"],ls="--",lw=2.2,label=f"Width growth ↔ (+{WIDTH_DELTA} neurons) ×{n_w}"),
    Line2D([0],[0],color=C["depth"],ls="--",lw=2.2,label=f"Depth growth ↕ (new layer) ×{n_d}"),
],loc="lower center",ncol=2,fontsize=9,facecolor=PBG,labelcolor="white",framealpha=0.78,bbox_to_anchor=(0.5,0.0))
p3 = os.path.join(tmpdir, "s3.png")
plt.savefig(p3, dpi=130, bbox_inches="tight", facecolor=BG); plt.close()
strip_paths.append(p3)

# Stitch
imgs   = [Image.open(p) for p in strip_paths]
W      = max(im.width for im in imgs)
scaled = [im.resize((W, int(im.height*W/im.width)), Image.LANCZOS) for im in imgs]
canvas = Image.new("RGB", (W, sum(s.height for s in scaled)), (13,13,28))
y = 0
for s in scaled: canvas.paste(s,(0,y)); y+=s.height
out = "neuroplasticity_cifar10_results.png"
canvas.save(out, quality=95)
print(f"\nFigure saved → {out}")

# In Colab: display inline
# from IPython.display import Image as IPImage; IPImage(out)
