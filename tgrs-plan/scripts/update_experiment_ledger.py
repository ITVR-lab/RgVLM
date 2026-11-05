#!/usr/bin/env python3
"""Regenerate TGRS experiment ledger from logs/artifacts.

On GPU box:
  python3.11 /hy-tmp/tgrs-plan/scripts/update_experiment_ledger.py

Writes:
  /hy-tmp/tgrs-plan/docs/EXPERIMENT_LEDGER.md
  /hy-tmp/tgrs-plan/artifacts/experiment_ledger.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

TZ = timezone(timedelta(hours=8))
ROOT = Path(os.environ.get("TGRS_ROOT", "/hy-tmp/tgrs-plan"))
LOGS = Path(os.environ.get("TGRS_LOGS", "/hy-tmp/logs"))
ART = ROOT / "artifacts"
DOCS = ROOT / "docs"
OUT_MD = DOCS / "EXPERIMENT_LEDGER.md"
OUT_JSON = ART / "experiment_ledger.json"

GATE1 = 89.178
PUBLIC_UNET = 91.42
BEST_SINGLE_BASELINE = 91.759


def now() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M Asia/Shanghai")


def parse_train_log(log: Path):
    if not log.is_file():
        return None
    t = log.read_text(errors="ignore")
    tests = re.findall(
        r"\[test\] ------- IoU --------\s*\n\s*Not Water\s+([0-9.]+)\s*\n\s*Water\s+([0-9.]+)\s*\n\s*\[test\]-------------------\s*\n\s*\[test\] Mean\s+([0-9.]+)",
        t,
    )
    means = re.findall(r"\[test\] ------- IoU --------[\s\S]*?\[test\] Mean\s+([0-9.]+)", t)
    vals = re.findall(r"\[val\] ------- IoU --------[\s\S]*?\[val\] Mean\s+([0-9.]+)", t)
    eps = re.findall(r"Starting epoch (\d+)", t)
    if tests:
        nw, water, mean = map(float, tests[-1])
    elif means:
        nw = water = None
        mean = float(means[-1])
    else:
        mean = nw = water = None
    return {
        "name": log.stem,
        "kind": "train",
        "test_miou": mean,
        "test_water": water,
        "test_notwater": nw,
        "best_val": max(map(float, vals)) if vals else None,
        "last_epoch": int(eps[-1]) if eps else None,
        "done": mean is not None or "[GGU] done" in t,
        "log": str(log),
        "mtime": datetime.fromtimestamp(log.stat().st_mtime, TZ).isoformat(timespec="minutes"),
    }


def parse_artifacts():
    rows = []
    if not ART.is_dir():
        return rows
    for p in sorted(ART.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        if "metrics" in d and isinstance(d["metrics"], dict) and "mIoU" in d["metrics"]:
            iou = d["metrics"].get("IoU") or []
            rows.append(
                {
                    "name": p.stem,
                    "kind": "ensemble",
                    "test_miou": float(d["metrics"]["mIoU"]),
                    "test_water": float(iou[1]) if len(iou) > 1 else None,
                    "notes": f"n={d.get('n_models')}",
                    "path": str(p),
                }
            )
        elif "baseline" in d and isinstance(d["baseline"], dict):
            iou = d["baseline"].get("IoU") or []
            rows.append(
                {
                    "name": p.stem,
                    "kind": "tta",
                    "test_miou": float(d["baseline"].get("mIoU")),
                    "tta_miou": float((d.get("tta") or {}).get("mIoU")) if d.get("tta") else None,
                    "test_water": float(iou[1]) if len(iou) > 1 else None,
                    "path": str(p),
                }
            )
        elif "test_mean" in d:
            rows.append(
                {
                    "name": p.stem,
                    "kind": "lock",
                    "test_miou": float(d["test_mean"]),
                    "tta_miou": float(d["hv_tta_mean"]) if d.get("hv_tta_mean") is not None else None,
                    "path": str(p),
                    "notes": d.get("recipe"),
                }
            )
        elif "verdict" in d:
            rows.append(
                {
                    "name": p.stem,
                    "kind": "verdict",
                    "test_miou": float(d["test"]) if d.get("test") is not None else None,
                    "notes": d.get("verdict"),
                    "path": str(p),
                }
            )
    return rows


def classify(name: str) -> str:
    n = name.lower()
    if "ensemble" in n:
        return "Ensemble"
    if n.startswith("tta_") or "_tta" in n or n.endswith("_hv") or "hvms" in n:
        return "TTA / eval"
    if n.startswith("gate1") or "caag" in n or "cafe" in n or "flora" in n:
        return "DOFA / CAAG track"
    if "scrs" in n:
        return "SCRS"
    if "spire" in n or "hydro_" in n or "hydrohard" in n or n.startswith("ggu"):
        return "Innovation arms"
    if "unet" in n or n.startswith("r34"):
        return "UNet family"
    return "Other"


def verdict_row(miou):
    if miou is None:
        return "pending"
    if miou >= 92.5:
        return "TARGET"
    if miou >= 92.0:
        return "PROMOTE"
    if miou >= BEST_SINGLE_BASELINE - 0.05:
        return "NEAR_SOTA"
    if miou >= 91.5:
        return "competitive"
    if miou >= GATE1:
        return "above_Gate1"
    return "KILL/weak"


def fmt(x, nd=3):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def gpu_disk():
    try:
        gpu = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader",
            shell=True,
        ).decode().strip()
    except Exception:
        gpu = "n/a"
    try:
        disk = subprocess.check_output("df -h /hy-tmp | tail -1", shell=True).decode().strip()
    except Exception:
        disk = "n/a"
    try:
        procs = subprocess.check_output(
            "pgrep -af 'queue_v2|pangaea/run|torchrun' | grep -v pgrep | head -8 || true",
            shell=True,
        ).decode().strip() or "(idle)"
    except Exception:
        procs = "n/a"
    return gpu, disk, procs


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    train_rows = []
    if LOGS.is_dir():
        for log in sorted(LOGS.glob("*.log")):
            info = parse_train_log(log)
            if info:
                train_rows.append(info)

    arts = parse_artifacts()
    gpu, disk, procs = gpu_disk()

    done = [r for r in train_rows if r.get("test_miou") is not None]
    pending = [r for r in train_rows if r.get("test_miou") is None and r.get("last_epoch")]
    done_sorted = sorted(done, key=lambda r: -r["test_miou"])

    ens = [a for a in arts if a["kind"] == "ensemble" and a.get("test_miou") is not None]
    best_ens = max(ens, key=lambda a: a["test_miou"]) if ens else None
    best_single = next((r for r in done_sorted if "unet_d4_160ep_seed234" == r["name"]), None)
    if best_single is None and done_sorted:
        # fallback: best non-tta train-looking name
        for r in done_sorted:
            if not r["name"].startswith("tta_") and "ensemble" not in r["name"]:
                best_single = r
                break

    lines = []
    lines += [
        "# TGRS / Sen1Floods11 Experiment Ledger",
        "",
        f"**Updated:** {now()}  ",
        f"**Root:** `{ROOT}`  ",
        "**Task:** Sen1Floods11 semantic segmentation · S2-only default · test mIoU",
        "",
        "> Auto-generated. Re-run `scripts/update_experiment_ledger.py` after each experiment.",
        "",
        "---",
        "",
        "## 0. Machine status",
        "",
        f"- **GPU:** `{gpu}`",
        f"- **Disk:** `{disk}`",
        "- **Procs:**",
        "```",
        procs,
        "```",
        "",
        "---",
        "",
        "## 1. Highlights",
        "",
        f"- Gate1 DOFA 3-seed lock: **{GATE1}**",
        f"- Public UNet approx: **{PUBLIC_UNET}**",
        f"- Best single (preferred): **{fmt(best_single['test_miou'] if best_single else None)}** (`{best_single['name'] if best_single else '—'}`)",
        f"- Best ensemble artifact: **{fmt(best_ens['test_miou'] if best_ens else None)}** (`{best_ens['name'] if best_ens else '—'}`)",
        f"- Parsed train/eval logs with test: **{len(done_sorted)}**",
        f"- In-progress logs: **{len(pending)}**",
        "",
        "### Kill / promote gates",
        "",
        "| Gate | Rule |",
        "|---|---|",
        "| KILL | test < 91.50 |",
        "| PROMOTE | test ≥ 92.0 |",
        "| TARGET | ≥ 92.5 |",
        "",
        "---",
        "",
        "## 2. Top train/eval results",
        "",
        "| test mIoU | Water | best val | Family | Exp | mtime | Verdict |",
        "|---:|---:|---:|---|---|---|---|",
    ]
    for r in done_sorted[:80]:
        lines.append(
            f"| {fmt(r['test_miou'])} | {fmt(r['test_water'])} | {fmt(r['best_val'])} | "
            f"{classify(r['name'])} | `{r['name']}` | {r['mtime']} | {verdict_row(r['test_miou'])} |"
        )
    if len(done_sorted) > 80:
        lines.append(f"| … | | | | ({len(done_sorted)-80} more in JSON) | | |")

    lines += ["", "## 3. Artifact ensembles / locks / TTA", ""]
    lines += ["| name | kind | baseline/test | TTA | notes |", "|---|---|---:|---:|---|"]
    for a in sorted(arts, key=lambda x: (-(x.get("test_miou") or -1), x["name"])):
        lines.append(
            f"| `{a['name']}` | {a['kind']} | {fmt(a.get('test_miou'))} | {fmt(a.get('tta_miou'))} | {a.get('notes') or ''} |"
        )

    if pending:
        lines += ["", "## 4. In progress", "", "| Exp | last epoch | mtime |", "|---|---:|---|"]
        for r in pending:
            lines.append(f"| `{r['name']}` | {r.get('last_epoch')} | {r.get('mtime')} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Next actions",
        "",
        "1. Do not revive CAAG-mix / SPIRe / HydroEdge as mainline.",
        "2. Watch `queue_v2.sh`; refresh ensemble after new seeds.",
        "3. Aim ≥92.5 via diverse ensemble → KD, or data-side hard mining.",
        "4. Keep `/hy-tmp` free ≥8GB before new runs.",
        "",
        f"## Changelog",
        "",
        f"- {now()}: ledger regenerated ({len(done_sorted)} test rows).",
        "",
    ]

    OUT_MD.write_text("\n".join(lines))
    payload = {
        "updated": now(),
        "root": str(ROOT),
        "gpu": gpu,
        "disk": disk,
        "procs": procs,
        "n_done": len(done_sorted),
        "best_single": best_single,
        "best_ensemble": best_ens,
        "train_results": done_sorted,
        "pending": pending,
        "artifacts": arts,
        "gates": {"kill": 91.5, "promote": 92.0, "target": 92.5, "gate1": GATE1},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print("WROTE", OUT_MD)
    print("WROTE", OUT_JSON)


if __name__ == "__main__":
    main()
