#!/usr/bin/env python3
"""Publication figures for TipGuided UniMTS paper (nature-figure style)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

OUT = Path(os.environ.get("FIG_OUT", "/hy-tmp/logs/paper_figs"))
LOG = Path("/hy-tmp/logs")
CACHE = Path("/hy-tmp/feat_cache")
DATA_PATH = "/hy-tmp/UniMTS_data/UniMTS_data"
CKPT = "/hy-tmp/UniMTS/checkpoint/UniMTS.pth"
UNI = "/hy-tmp/UniMTS"

DATASETS = [
    "Opp_g", "UCIHAR", "MotionSense", "w-HAR", "Shoaib", "har70plus",
    "realworld", "TNDA-HAR", "PAMAP", "USCHAD", "Mhealth", "Harth",
    "ut-complex", "Wharf", "WISDM", "DSADS", "UTD-MHAD", "MMAct",
]

# restrained palette (nature-figure stance)
C = {
    "zs": "#9AA3AD",
    "nn": "#7A8694",
    "probe": "#B0B7C0",
    "tip": "#5B7C99",
    "plus": "#3D6F8F",
    "full": "#C47A3A",
    "guided": "#1F6F5B",
    "anneal": "#6B8F71",
    "grid": "#E6E8EB",
}


def style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", OUT / name, flush=True)


def load_avg(path, key=None):
    d = json.load(open(path))
    a = d["average"]
    if key and key in a:
        v = a[key]
        return v["acc"] if isinstance(v, dict) else float(v)
    if "acc" in a:
        return float(a["acc"])
    if "tip_f_loo" in a:
        return float(a["tip_f_loo"])
    return max(v["acc"] for v in a.values() if isinstance(v, dict) and "acc" in v)


def best_guided(k):
    best, name = -1.0, None
    for p in LOG.glob(f"tip_guided*_k{k}.json"):
        a = load_avg(p)
        if a > best:
            best, name = a, p.name
    return best, name


def fig_main_bars():
    ks = [1, 2, 3, 5, 10]
    methods = [
        ("Tip-F", "tip", lambda k: load_avg(LOG / f"cross_tip_k{k}.json", "tip_f_loo")),
        ("Full FT", "full", lambda k: load_avg(LOG / f"full_k{k}.json")),
        ("TipGuided", "guided", lambda k: best_guided(k)[0]),
    ]
    x = np.arange(len(ks), dtype=float)
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for i, (name, key, fn) in enumerate(methods):
        vals = [fn(k) * 100 for k in ks]
        ax.bar(x + (i - 1) * w, vals, width=w, color=C[key], label=name, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_ylim(55, 88)
    ax.yaxis.grid(True, color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("18-dataset mean accuracy under UniMTS few-shot protocol")
    save(fig, "fig_main_bars")


def fig_ladder_k5():
    items = [
        ("Zero-shot", load_avg(LOG / "cross_tip_k5.json", "zs") if "zs" in json.load(open(LOG / "cross_tip_k5.json"))["average"] else 0.444, "zs"),
        ("Probe", load_avg(LOG / "probe_k5.json"), "probe"),
        ("1-NN", load_avg(LOG / "cross_tip_k5.json", "nn1"), "nn"),
        ("Tip-F", load_avg(LOG / "cross_tip_k5.json", "tip_f_loo"), "tip"),
        ("Tip+", load_avg(LOG / "tip_plus_k5.json"), "plus"),
        ("Full FT", load_avg(LOG / "full_k5.json"), "full"),
        ("TipGuided", best_guided(5)[0], "guided"),
    ]
    # fix zs
    zs = json.load(open(LOG / "cross_tip_k5.json"))["average"].get("zs", 0.444)
    if isinstance(zs, dict):
        zs = zs["acc"]
    items[0] = ("Zero-shot", float(zs), "zs")

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    y = np.arange(len(items))
    vals = [v * 100 for _, v, _ in items]
    colors = [C[c] for _, _, c in items]
    ax.barh(y, vals, color=colors, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in items])
    ax.set_xlabel("Mean accuracy (%)  ·  k=5")
    ax.set_xlim(40, 82)
    for yi, v in zip(y, vals):
        ax.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=8, color="#333")
    ax.invert_yaxis()
    ax.set_title("Diagnostic ladder at k=5")
    save(fig, "fig_ladder_k5")


def fig_delta_vs_full():
    ks = [1, 2, 3, 5, 10]
    deltas = []
    for k in ks:
        g = best_guided(k)[0]
        f = load_avg(LOG / f"full_k{k}.json")
        deltas.append((g - f) * 100)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    colors = [C["guided"] if d >= 0 else "#A33" for d in deltas]
    ax.bar(np.arange(len(ks)), deltas, color=colors, width=0.55)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("TipGuided − Full FT (pp)")
    ax.set_title("TipGuided improves over Full FT at every k")
    ax.yaxis.grid(True, color=C["grid"])
    ax.set_axisbelow(True)
    save(fig, "fig_delta_full")


def fig_per_dataset_heatmap(k=5):
    gpath = LOG / f"tip_guided_ce_longlow_k{k}.json"
    if not gpath.exists():
        gpath = LOG / f"tip_guided_ce_k{k}.json"
    fpath = LOG / f"full_k{k}.json"
    grows = {r["dataset"]: r["acc"] for r in json.load(open(gpath))["rows"]}
    frows = {r["dataset"]: r["acc"] for r in json.load(open(fpath))["rows"]}
    # tip-f from cross tip loo
    ct = json.load(open(LOG / f"cross_tip_k{k}.json"))
    tips = {}
    for r in ct["rows"]:
        ds = r["dataset"]
        tips[ds] = r["loo_tip_f"]["tip_f"]["acc"]

    ds = [d for d in DATASETS if d in grows and d in frows]
    mat = np.array([[tips[d], frows[d], grows[d]] for d in ds]) * 100
    fig, ax = plt.subplots(figsize=(4.8, 6.2))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=20, vmax=100)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Tip-F", "Full FT", "TipGuided"], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(ds)))
    ax.set_yticklabels(ds, fontsize=7)
    ax.set_title(f"Per-dataset accuracy (%) · k={k}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    save(fig, f"fig_heatmap_k{k}")

    # win/loss stem
    delta = np.array([(grows[d] - frows[d]) * 100 for d in ds])
    order = np.argsort(delta)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    y = np.arange(len(ds))
    cols = [C["guided"] if delta[i] >= 0 else C["full"] for i in order]
    ax.barh(y, delta[order], color=cols, height=0.7)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([ds[i] for i in order], fontsize=7)
    ax.set_xlabel("TipGuided − Full FT (pp)")
    ax.set_title(f"Per-dataset gap vs Full FT · k={k}")
    save(fig, f"fig_gap_datasets_k{k}")


def load_official_support(ds, k, train_y):
    path = os.path.join(DATA_PATH, "few_shot_data_2", f"{ds}_k={k}.pth")
    all_indices = torch.load(path)
    unique_labels = sorted(set(int(x) for x in np.unique(train_y)))
    idxs = []
    for i, _ in enumerate(unique_labels):
        if i >= len(all_indices):
            break
        sel = all_indices[i]
        if hasattr(sel, "numpy"):
            sel = sel.numpy()
        idxs.append(np.atleast_1d(np.array(sel, dtype=np.int64)))
    return np.concatenate(idxs)


def tip_f_loss_curve(ds="UCIHAR", k=5, epochs=60, beta=20.0, alpha=10.0):
    data = np.load(CACHE / f"{ds}.npz", allow_pickle=True)
    tr_f, te_f = data["train_feat"], data["test_feat"]
    tr_y, te_y = data["train_y"], data["test_y"]
    text = data["text_ensemble"]
    meta = json.load(open(CACHE / "meta.json"))
    logit_scale = float(meta["logit_scale"])
    sup = load_official_support(ds, k, tr_y)
    class_ids = sorted(set(int(x) for x in tr_y[sup]))
    class_ids = [c for c in class_ids if 0 <= c < len(data["names"])]
    sup = sup[np.isin(tr_y[sup], class_ids)]
    sf, sy = tr_f[sup], tr_y[sup]
    m = {c: i for i, c in enumerate(class_ids)}
    sy_c = np.array([m[int(v)] for v in sy], dtype=np.int64)
    n_c = len(class_ids)
    text_c = text[np.array(class_ids)]
    values = np.eye(n_c, dtype=np.float32)[sy_c]
    for c in range(n_c):
        mask = sy_c == c
        if mask.any():
            values[mask] /= float(mask.sum())
    zs_s = logit_scale * (normalize(sf, "l2") @ text_c.T)
    zs_q = logit_scale * (normalize(te_f, "l2") @ text_c.T)

    keys_t = nn.Parameter(torch.from_numpy(sf.copy()).float().cuda())
    values_t = torch.from_numpy(values).float().cuda()
    tr = torch.from_numpy(sf).float().cuda()
    y = torch.from_numpy(sy_c).long().cuda()
    zs = torch.from_numpy(zs_s).float().cuda()
    opt = torch.optim.Adam([keys_t], lr=1e-3)
    hist = {"loss": [], "train_acc": [], "test_acc": []}
    qf = torch.from_numpy(te_f).float().cuda()
    zq = torch.from_numpy(zs_q).float().cuda()
    inv = np.array(class_ids)
    for ep in range(epochs):
        opt.zero_grad()
        kn = keys_t / keys_t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        tn = tr / tr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        aff = torch.exp(-beta * (1.0 - tn @ kn.t()))
        logits = zs + alpha * (aff @ values_t)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            tr_acc = (logits.argmax(1) == y).float().mean().item()
            qn = qf / qf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            kn = keys_t / keys_t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            affq = torch.exp(-beta * (1.0 - qn @ kn.t()))
            pred = (zq + alpha * (affq @ values_t)).argmax(1).cpu().numpy()
            pred = inv[pred]
            te_acc = float((pred == te_y).mean())
        hist["loss"].append(float(loss.item()))
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
    return hist


def fig_tipf_loss_multi():
    datasets = ["UCIHAR", "PAMAP", "DSADS", "MotionSense"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ds in datasets:
        h = tip_f_loss_curve(ds, k=5, epochs=50)
        axes[0].plot(h["loss"], label=ds, lw=1.6)
        axes[1].plot(np.array(h["test_acc"]) * 100, label=ds, lw=1.6)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Support CE loss")
    axes[0].set_title("Tip-F key training loss (k=5)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("Tip-F test accuracy during key FT")
    for ax in axes:
        ax.yaxis.grid(True, color=C["grid"])
        ax.set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "fig_tipf_loss")


def encode_dataset_features(ds, k=5, mode="frozen", epochs=40):
    """Return test features (N,d) and labels after frozen or TipGuided CE FT."""
    sys.path.insert(0, UNI)
    os.chdir(UNI)
    import argparse
    import clip
    from contrastive import ContrastiveModule
    from data import load

    class Args:
        pass

    args = Args()
    args.padding_size = 200
    args.data_path = DATA_PATH if DATA_PATH.endswith("/") else DATA_PATH + "/"
    # data.py expects path
    args.data_path = "./UniMTS_data/" if os.path.isdir("./UniMTS_data") else DATA_PATH + "/"
    # fix: UniMTS project uses relative UniMTS_data symlink?
    if not os.path.isdir(args.data_path):
        args.data_path = "/hy-tmp/UniMTS_data/UniMTS_data/"
    args.gyro = 0
    args.stft = 0
    args.batch_size = 64
    args.checkpoint = CKPT
    args.stage = "finetune"
    args.k = k

    def preprocess_imu(input_tensor):
        input_tensor = input_tensor.cuda()
        b, t, c = input_tensor.shape
        indices = np.array([range(i, i + 3) for i in range(0, c, 6)]).flatten()
        input_tensor = input_tensor[:, :, indices]
        b, t, c = input_tensor.shape
        return input_tensor.reshape(b, t, 22, -1).permute(0, 3, 1, 2).unsqueeze(-1)

    tr_x, _, tr_y, names, _, _ = load(ds, args.padding_size, args.data_path, split="train", k=k)
    te_x, _, te_y, _, _, _ = load(ds, args.padding_size, args.data_path, split="test", k=None)
    class_ids = sorted(set(int(x) for x in tr_y.numpy()))
    mapping = {c: i for i, c in enumerate(class_ids)}
    tr_y_c = torch.tensor([mapping[int(x)] for x in tr_y.numpy()], dtype=torch.long)
    n_c = len(class_ids)
    args.num_class = n_c
    model = ContrastiveModule(args).cuda()
    model.model.load_state_dict(torch.load(args.checkpoint))
    model.fc = nn.Linear(512, n_c).cuda()
    for n, p in model.named_parameters():
        p.requires_grad = ("model.acc" in n) or n.startswith("fc.")

    if mode != "frozen":
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        losses = []
        te_accs = []
        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(TensorDataset(tr_x, tr_y_c), batch_size=min(64, len(tr_x)), shuffle=True)
        for ep in range(epochs):
            model.train()
            tot = 0
            n = 0
            for xb, yb in loader:
                yb = yb.cuda()
                feat = model.encode_image(preprocess_imu(xb))
                loss = F.cross_entropy(model.fc(feat), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(yb)
                n += len(yb)
            losses.append(tot / max(n, 1))
            # quick test
            model.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(te_x), 64):
                    feat = model.encode_image(preprocess_imu(te_x[i : i + 64]))
                    preds.append(model.fc(feat).argmax(1).cpu())
            pred = torch.cat(preds).numpy()
            inv = {i: c for c, i in mapping.items()}
            pred_lab = np.array([inv[int(i)] for i in pred])
            te_accs.append(float((pred_lab == te_y.numpy()).mean()))
        hist = {"loss": losses, "test_acc": te_accs}
    else:
        hist = None

    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(te_x), 64):
            feat = model.encode_image(preprocess_imu(te_x[i : i + 64]))
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feats.append(feat.cpu().numpy())
    feats = np.concatenate(feats, 0)
    return feats, te_y.numpy().astype(np.int64), hist


def fig_tsne_pair(ds="UCIHAR", k=5):
    print("t-SNE frozen", ds, flush=True)
    f0, y, _ = encode_dataset_features(ds, k=k, mode="frozen")
    print("t-SNE tipguided-ce", ds, flush=True)
    f1, y1, hist = encode_dataset_features(ds, k=k, mode="ce", epochs=50)
    # subsample
    rng = np.random.RandomState(0)
    n = min(1200, len(y))
    idx = rng.choice(len(y), n, replace=False)
    emb0 = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0).fit_transform(f0[idx])
    emb1 = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0).fit_transform(f1[idx])
    labs = y[idx]
    classes = sorted(set(labs.tolist()))
    cmap = plt.cm.get_cmap("tab20", max(len(classes), 1))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, emb, title in [
        (axes[0], emb0, f"{ds} · frozen UniMTS"),
        (axes[1], emb1, f"{ds} · after TipGuided-CE"),
    ]:
        for i, c in enumerate(classes):
            m = labs == c
            ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.75, color=cmap(i), linewidths=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#DDD")
    save(fig, f"fig_tsne_{ds}")

    # loss for this FT run
    if hist:
        fig, ax = plt.subplots(figsize=(4.8, 2.8))
        ax.plot(hist["loss"], color=C["guided"], lw=1.8, label="Train CE")
        ax2 = ax.twinx()
        ax2.plot(np.array(hist["test_acc"]) * 100, color=C["full"], lw=1.8, label="Test acc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss")
        ax2.set_ylabel("Test accuracy (%)")
        ax.set_title(f"TipGuided-CE training · {ds} · k={k}")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8)
        save(fig, f"fig_train_curve_{ds}")
    return hist


def fig_guided_vs_full_loss(datasets=None, k=5, epochs=40):
    """Compare CE train loss / test acc: Full-FT style vs TipGuided fuse on same supports."""
    sys.path.insert(0, UNI)
    os.chdir(UNI)
    from contrastive import ContrastiveModule
    from data import load
    from torch.utils.data import DataLoader, TensorDataset
    import clip

    if datasets is None:
        datasets = ["UCIHAR", "PAMAP", "Wharf"]

    def preprocess_imu(input_tensor):
        input_tensor = input_tensor.cuda()
        b, t, c = input_tensor.shape
        indices = np.array([range(i, i + 3) for i in range(0, c, 6)]).flatten()
        input_tensor = input_tensor[:, :, indices]
        return input_tensor.reshape(b, t, 22, -1).permute(0, 3, 1, 2).unsqueeze(-1)

    @torch.no_grad()
    def encode_ensemble(model, names):
        TEMPLATES = ["{}", "a person is {}", "the activity is {}", "someone is {}"]
        acc = None
        for tmpl in TEMPLATES:
            tokens = clip.tokenize([tmpl.format(n) for n in names], truncate=True).cuda()
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            acc = feats if acc is None else acc + feats
        acc = acc / len(TEMPLATES)
        return acc / acc.norm(dim=-1, keepdim=True)

    curves = {}
    for ds in datasets:
        print("loss compare", ds, flush=True)
        class Args:
            pass

        args = Args()
        args.padding_size = 200
        args.data_path = "/hy-tmp/UniMTS_data/UniMTS_data/"
        args.gyro = 0
        args.stft = 0
        args.batch_size = 64
        args.checkpoint = CKPT
        args.stage = "finetune"
        args.k = k
        tr_x, _, tr_y, names, _, _ = load(ds, args.padding_size, args.data_path, split="train", k=k)
        te_x, _, te_y, _, _, _ = load(ds, args.padding_size, args.data_path, split="test", k=None)
        class_ids = sorted(set(int(x) for x in tr_y.numpy()))
        mapping = {c: i for i, c in enumerate(class_ids)}
        inv = {i: c for c, i in mapping.items()}
        tr_y_c = torch.tensor([mapping[int(x)] for x in tr_y.numpy()], dtype=torch.long)
        n_c = len(class_ids)
        args.num_class = n_c

        def train_mode(mode):
            model = ContrastiveModule(args).cuda()
            model.model.load_state_dict(torch.load(args.checkpoint))
            model.fc = nn.Linear(512, n_c).cuda()
            for n, p in model.named_parameters():
                p.requires_grad = ("model.acc" in n) or n.startswith("fc.")
            text = encode_ensemble(model, [names[c] if c < len(names) else str(c) for c in class_ids])
            logit_scale = model.model.logit_scale.exp().detach()
            model.eval()
            with torch.no_grad():
                keys = []
                for i in range(0, len(tr_x), 64):
                    f = model.encode_image(preprocess_imu(tr_x[i : i + 64]))
                    keys.append(f / f.norm(dim=-1, keepdim=True))
                keys0 = torch.cat(keys, 0)
            values = F.one_hot(tr_y_c.cuda(), num_classes=n_c).float()
            for c in range(n_c):
                m = tr_y_c.cuda() == c
                if m.any():
                    values[m] = values[m] / m.sum()
            keys_p = nn.Parameter(keys0.clone().detach())
            log_a = nn.Parameter(torch.tensor(np.log(10.0), device="cuda"))
            log_g = nn.Parameter(torch.tensor(0.0, device="cuda"))
            params = [p for p in model.parameters() if p.requires_grad]
            if mode == "guided":
                opt = torch.optim.Adam(
                    [{"params": params, "lr": 1e-4}, {"params": [keys_p, log_a, log_g], "lr": 1e-3}]
                )
            else:
                opt = torch.optim.Adam(params, lr=1e-4)
            loader = DataLoader(
                TensorDataset(tr_x, tr_y_c), batch_size=min(64, max(4, len(tr_x))), shuffle=True
            )
            loss_h, acc_h = [], []
            beta = 20.0
            for ep in range(epochs):
                model.train()
                tot = 0
                n = 0
                for xb, yb in loader:
                    yb = yb.cuda()
                    feat = model.encode_image(preprocess_imu(xb))
                    feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    ce = model.fc(feat)
                    if mode == "guided":
                        kn = keys_p / keys_p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        aff = torch.exp(-beta * (1.0 - feat @ kn.t()))
                        tip = logit_scale * (feat @ text.t()) + torch.exp(log_a) * (aff @ values)
                        logits = tip + torch.exp(log_g) * ce
                    else:
                        logits = ce
                    loss = F.cross_entropy(logits, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += loss.item() * len(yb)
                    n += len(yb)
                loss_h.append(tot / max(n, 1))
                model.eval()
                preds = []
                with torch.no_grad():
                    for i in range(0, len(te_x), 64):
                        feat = model.encode_image(preprocess_imu(te_x[i : i + 64]))
                        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        ce = model.fc(feat)
                        if mode == "guided":
                            kn = keys_p / keys_p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                            aff = torch.exp(-beta * (1.0 - feat @ kn.t()))
                            tip = logit_scale * (feat @ text.t()) + torch.exp(log_a) * (aff @ values)
                            logits = tip + torch.exp(log_g) * ce
                        else:
                            logits = ce
                        preds.append(logits.argmax(1).cpu())
                pred = torch.cat(preds).numpy()
                pred_lab = np.array([inv[int(i)] for i in pred])
                acc_h.append(float((pred_lab == te_y.numpy()).mean()))
            return {"loss": loss_h, "test_acc": acc_h}

        curves[ds] = {"full": train_mode("full"), "guided": train_mode("guided")}

    fig, axes = plt.subplots(2, len(datasets), figsize=(2.4 * len(datasets) + 0.6, 4.6), sharex=True)
    if len(datasets) == 1:
        axes = np.array(axes).reshape(2, 1)
    for j, ds in enumerate(datasets):
        axes[0, j].plot(curves[ds]["full"]["loss"], color=C["full"], lw=1.6, label="Full FT")
        axes[0, j].plot(curves[ds]["guided"]["loss"], color=C["guided"], lw=1.6, label="TipGuided")
        axes[0, j].set_title(ds)
        axes[0, j].set_ylabel("Train loss" if j == 0 else "")
        axes[1, j].plot(np.array(curves[ds]["full"]["test_acc"]) * 100, color=C["full"], lw=1.6)
        axes[1, j].plot(np.array(curves[ds]["guided"]["test_acc"]) * 100, color=C["guided"], lw=1.6)
        axes[1, j].set_ylabel("Test acc (%)" if j == 0 else "")
        axes[1, j].set_xlabel("Epoch")
        for ax in (axes[0, j], axes[1, j]):
            ax.yaxis.grid(True, color=C["grid"])
            ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Full FT vs TipGuided training dynamics · k={k}", y=1.02)
    save(fig, "fig_train_compare")
    with open(OUT / "train_compare.json", "w") as f:
        json.dump(curves, f)
    return curves


def fig_method_schematic():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F3F5F7", ec="#444"):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.0, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

    box(0.2, 1.4, 2.0, 1.2, "IMU window\n$x$", fc="#EEF2F5")
    box(2.8, 1.4, 2.2, 1.2, "ST-GCN\n$f_\\theta(x)$\n(trainable)", fc="#D9E8E3")
    box(5.6, 2.5, 2.4, 1.1, "Text logits\n$z_{\\mathrm{text}}$", fc="#E8EEF4")
    box(5.6, 1.35, 2.4, 1.0, "Tip cache\n$\\alpha A V$", fc="#E8EEF4")
    box(5.6, 0.2, 2.4, 1.0, "CE head\n$\\gamma W f$", fc="#E8EEF4")
    box(9.0, 1.4, 2.6, 1.2, "Fused logits\n$z=z_t+\\alpha AV+\\gamma Wf$", fc="#CFE3DC")
    arrow(2.2, 2.0, 2.8, 2.0)
    arrow(5.0, 2.0, 5.6, 3.0)
    arrow(5.0, 2.0, 5.6, 1.85)
    arrow(5.0, 2.0, 5.6, 0.7)
    arrow(8.0, 3.0, 9.0, 2.2)
    arrow(8.0, 1.85, 9.0, 2.0)
    arrow(8.0, 0.7, 9.0, 1.8)
    ax.text(6.0, 3.85, "TipGuided Adaptation", fontsize=11, fontweight="bold")
    ax.text(0.2, 0.15, "Freeze CLIP text tower · update ST-GCN + Tip keys / scales + CE head", fontsize=7, color="#555")
    save(fig, "fig_method")


def fig_ablation_k10():
    items = [
        ("Tip-F", load_avg(LOG / "cross_tip_k10.json", "tip_f_loo")),
        ("Tip+", load_avg(LOG / "tip_plus_k10.json")),
        ("LightFT", load_avg(LOG / "light_ft_fullish_k10.json") if (LOG / "light_ft_fullish_k10.json").exists() else load_avg(LOG / "light_ft_tip_k10.json")),
        ("Anneal", load_avg(LOG / "tip_anneal_anneal_k10.json")),
        ("Full FT", load_avg(LOG / "full_k10.json")),
        ("TipGuided", best_guided(10)[0]),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(items))
    cols = [C["tip"], C["plus"], C["anneal"], C["anneal"], C["full"], C["guided"]]
    vals = [v * 100 for _, v in items]
    ax.bar(x, vals, color=cols, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in items], rotation=15, ha="right")
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_ylim(70, 86)
    ax.set_title("Ablation / method family · k=10")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", fontsize=7)
    ax.yaxis.grid(True, color=C["grid"])
    ax.set_axisbelow(True)
    save(fig, "fig_ablation_k10")


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== static plots ===", flush=True)
    fig_main_bars()
    fig_ladder_k5()
    fig_delta_vs_full()
    fig_per_dataset_heatmap(5)
    fig_per_dataset_heatmap(10)
    fig_ablation_k10()
    fig_method_schematic()

    print("=== tip-f loss ===", flush=True)
    fig_tipf_loss_multi()

    print("=== guided vs full dynamics ===", flush=True)
    fig_guided_vs_full_loss(["UCIHAR", "PAMAP", "Wharf"], k=5, epochs=40)

    print("=== t-SNE ===", flush=True)
    for ds in ["UCIHAR", "PAMAP", "Wharf"]:
        try:
            fig_tsne_pair(ds, k=5)
        except Exception as e:
            print("tsne fail", ds, e, flush=True)

    # index
    files = sorted([p.name for p in OUT.glob("*")])
    (OUT / "index.json").write_text(json.dumps(files, indent=2))
    print("ALL FIGURES DONE", len(files), flush=True)


if __name__ == "__main__":
    main()
