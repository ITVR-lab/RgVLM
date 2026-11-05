# TGRS / Sen1Floods11 Experiment Ledger

**Updated:** 2026-11-05 12:12 Asia/Shanghai  
**Root:** `/hy-tmp/tgrs-plan`  
**Task:** Sen1Floods11 semantic segmentation · S2-only default · test mIoU

> Auto-generated. Re-run `scripts/update_experiment_ledger.py` after each experiment.

---

## 0. Machine status

- **GPU:** `n/a`
- **Disk:** `/dev/nvme0n1     50G   13G   38G  25% /hy-tmp`
- **Procs:**
```
(idle)
```

---

## 1. Highlights

- Gate1 DOFA 3-seed lock: **89.178**
- Public UNet approx: **91.42**
- Best single (preferred): **91.759** (`unet_d4_160ep_seed234`)
- Best ensemble artifact: **91.925** (`ensemble_diverse_best3`)
- Parsed train/eval logs with test: **104**
- In-progress logs: **6**

### Kill / promote gates

| Gate | Rule |
|---|---|
| KILL | test < 91.50 |
| PROMOTE | test ≥ 92.0 |
| TARGET | ≥ 92.5 |

---

## 2. Top train/eval results

| test mIoU | Water | best val | Family | Exp | mtime | Verdict |
|---:|---:|---:|---|---|---|---|
| 91.842 | 85.796 | — | TTA / eval | `tta_unet_s2_seed42_hvms` | 2026-11-03T19:49+08:00 | NEAR_SOTA |
| 91.826 | 85.753 | — | TTA / eval | `tta_best_rerun` | 2026-11-05T11:46+08:00 | NEAR_SOTA |
| 91.826 | 85.753 | — | TTA / eval | `tta_unet_d4_160ep_seed234_hv` | 2026-11-04T15:08+08:00 | NEAR_SOTA |
| 91.822 | 85.765 | — | TTA / eval | `tta_unet_s2_seed42_hv` | 2026-11-03T19:48+08:00 | NEAR_SOTA |
| 91.782 | 85.690 | — | TTA / eval | `tta_unet_d4_160ep_seed42_hv` | 2026-11-04T15:35+08:00 | NEAR_SOTA |
| 91.759 | 85.638 | — | TTA / eval | `tta_unet_d4_160ep_seed234_hvms` | 2026-11-04T15:09+08:00 | NEAR_SOTA |
| 91.759 | 85.638 | 91.737 | UNet family | `unet_d4_160ep_seed234` | 2026-11-03T20:18+08:00 | NEAR_SOTA |
| 91.735 | 85.611 | 91.762 | UNet family | `unet_d4_cedice_seed234` | 2026-11-05T00:47+08:00 | NEAR_SOTA |
| 91.717 | 85.583 | 91.099 | UNet family | `unet_s2_seed42` | 2026-11-03T19:27+08:00 | NEAR_SOTA |
| 91.685 | 85.523 | 91.754 | UNet family | `unet_d4_160ep_seed42` | 2026-11-04T15:35+08:00 | competitive |
| 91.558 | 85.317 | — | TTA / eval | `tta_unet_s2_seed123_hv` | 2026-11-03T19:49+08:00 | competitive |
| 91.556 | 85.309 | — | TTA / eval | `tta_unet_s2_seed123_hvms` | 2026-11-03T19:50+08:00 | competitive |
| 91.544 | 85.292 | 91.222 | UNet family | `unet_s2_seed123` | 2026-11-03T19:40+08:00 | competitive |
| 91.527 | 85.246 | 91.186 | UNet family | `unet_d4_seed123` | 2026-11-03T20:54+08:00 | competitive |
| 91.475 | 85.168 | 91.403 | UNet family | `unet_d4_lovasz_seed234` | 2026-11-03T19:53+08:00 | above_Gate1 |
| 91.438 | 85.109 | 91.206 | UNet family | `unet_mm_d4_seed123` | 2026-11-03T21:32+08:00 | above_Gate1 |
| 91.399 | 85.026 | 91.034 | UNet family | `unet_mm_d4_seed234` | 2026-11-03T21:09+08:00 | above_Gate1 |
| 91.269 | 84.831 | 90.670 | UNet family | `r34_d4_seed234` | 2026-11-03T20:30+08:00 | above_Gate1 |
| 91.247 | 84.776 | 90.972 | UNet family | `unet_d4_seed234` | 2026-11-03T18:56+08:00 | above_Gate1 |
| 91.226 | 84.724 | 90.909 | UNet family | `unet_d4_seed42` | 2026-11-03T20:43+08:00 | above_Gate1 |
| 91.123 | 84.526 | 90.642 | Innovation arms | `ggu_base_seed234` | 2026-11-03T18:28+08:00 | above_Gate1 |
| 91.013 | 84.314 | — | TTA / eval | `tta_unet_s2_seed234_hvms` | 2026-11-03T19:48+08:00 | above_Gate1 |
| 91.004 | 84.304 | — | TTA / eval | `tta_unet_s2_seed234_hv` | 2026-11-03T19:47+08:00 | above_Gate1 |
| 90.988 | 84.294 | 91.871 | UNet family | `unet_recipe_seed42` | 2026-11-03T22:19+08:00 | above_Gate1 |
| 90.943 | 84.198 | 90.833 | UNet family | `unet_s2_seed234` | 2026-11-03T18:13+08:00 | above_Gate1 |
| 90.905 | 84.145 | 91.934 | UNet family | `unet_d4_200ep_seed234` | 2026-11-05T00:24+08:00 | above_Gate1 |
| 90.886 | 84.109 | 92.002 | UNet family | `unet_recipe_seed234` | 2026-11-03T21:57+08:00 | above_Gate1 |
| 90.878 | 84.115 | 90.686 | Innovation arms | `ggu_d4_seed234` | 2026-11-03T19:15+08:00 | above_Gate1 |
| 90.874 | 84.112 | 90.502 | Innovation arms | `ggu_idx_seed234` | 2026-11-03T18:43+08:00 | above_Gate1 |
| 90.813 | 83.970 | — | TTA / eval | `tta_unet_d4_160ep_seed123_hv` | 2026-11-04T15:58+08:00 | above_Gate1 |
| 90.761 | 83.890 | 91.900 | UNet family | `unet_recipe_seed123` | 2026-11-03T22:42+08:00 | above_Gate1 |
| 90.759 | 83.878 | 91.662 | UNet family | `unet_d4_160ep_seed123` | 2026-11-04T15:58+08:00 | above_Gate1 |
| 90.628 | 83.652 | 92.277 | UNet family | `unet_recipe240_seed234` | 2026-11-03T23:16+08:00 | above_Gate1 |
| 90.576 | 83.584 | 91.094 | UNet family | `unet_mm_d4_seed42` | 2026-11-03T21:21+08:00 | above_Gate1 |
| 90.516 | 83.477 | 92.012 | Innovation arms | `spire_unet_seed234` | 2026-11-04T22:35+08:00 | above_Gate1 |
| 90.369 | 83.276 | 88.732 | Innovation arms | `hydro_edge_200ep_seed234` | 2026-11-05T01:39+08:00 | above_Gate1 |
| 90.217 | 83.053 | 88.888 | Innovation arms | `hydro_edge_cedice_seed234` | 2026-11-05T01:10+08:00 | above_Gate1 |
| 89.964 | 82.545 | 88.691 | Innovation arms | `hydro_edge_seed234` | 2026-11-04T23:14+08:00 | above_Gate1 |
| 89.773 | 82.218 | — | TTA / eval | `scrs_tta_seed234` | 2026-11-03T13:57+08:00 | above_Gate1 |
| 89.769 | 82.193 | — | TTA / eval | `scrs_tta_seed42` | 2026-11-03T13:59+08:00 | above_Gate1 |
| 89.724 | 82.145 | — | TTA / eval | `scrs_ms_tta_all` | 2026-11-03T17:08+08:00 | above_Gate1 |
| 89.599 | 81.901 | — | TTA / eval | `gate1_tta_seed234` | 2026-11-02T23:04+08:00 | above_Gate1 |
| 89.571 | 81.789 | — | TTA / eval | `tta_hydrohard_seed234_hv` | 2026-11-04T23:15+08:00 | above_Gate1 |
| 89.568 | 81.850 | 89.023 | SCRS | `scrs_seed42` | 2026-11-02T23:42+08:00 | above_Gate1 |
| 89.537 | 81.833 | — | TTA / eval | `scrs_tta_seed123` | 2026-11-03T13:59+08:00 | above_Gate1 |
| 89.497 | 81.778 | — | TTA / eval | `gate1_tta_seed123` | 2026-11-02T23:05+08:00 | above_Gate1 |
| 89.488 | 81.719 | 88.918 | SCRS | `scrs_fullft_ce_seed234` | 2026-11-03T17:28+08:00 | above_Gate1 |
| 89.484 | 81.727 | 89.191 | SCRS | `scrs_seed234` | 2026-11-02T23:33+08:00 | above_Gate1 |
| 89.472 | 81.688 | 88.924 | Innovation arms | `hydro_wide_seed234` | 2026-11-04T23:55+08:00 | above_Gate1 |
| 89.369 | 81.512 | — | TTA / eval | `gate1_tta_seed42` | 2026-11-02T23:04+08:00 | above_Gate1 |
| 89.330 | 81.487 | 88.722 | SCRS | `scrs_partialft_l2_cedice_seed234` | 2026-11-03T17:14+08:00 | above_Gate1 |
| 89.314 | 81.427 | 88.961 | DOFA / CAAG track | `caag_full_pangaea_fttrue_seed234` | 2026-11-02T14:39+08:00 | above_Gate1 |
| 89.286 | 81.364 | — | DOFA / CAAG track | `gate1_eval_audit_seed234` | 2026-11-02T13:37+08:00 | above_Gate1 |
| 89.286 | 81.364 | 88.811 | DOFA / CAAG track | `gate1_s2_segdefault_80ep` | 2026-11-02T13:18+08:00 | above_Gate1 |
| 89.286 | 81.407 | 89.223 | SCRS | `scrs_seed123` | 2026-11-03T13:53+08:00 | above_Gate1 |
| 89.272 | 81.358 | 88.849 | DOFA / CAAG track | `caag_zeroinit_pangaea_seed234` | 2026-11-02T14:00+08:00 | above_Gate1 |
| 89.234 | 81.282 | 88.738 | DOFA / CAAG track | `caag_full_pangaea_seed42` | 2026-11-02T14:24+08:00 | above_Gate1 |
| 89.205 | 81.265 | 88.747 | DOFA / CAAG track | `caag_full_pangaea_fttrue_seed123` | 2026-11-02T14:53+08:00 | above_Gate1 |
| 89.193 | 81.219 | 88.815 | DOFA / CAAG track | `cafe_lite_seed234` | 2026-11-02T22:43+08:00 | above_Gate1 |
| 89.190 | 81.253 | 88.646 | DOFA / CAAG track | `gate1_s2_segdefault_seed123` | 2026-11-02T13:35+08:00 | above_Gate1 |
| 89.189 | 81.212 | 88.816 | DOFA / CAAG track | `caag_full_pangaea_seed234` | 2026-11-02T14:23+08:00 | above_Gate1 |
| 89.158 | 81.167 | 88.691 | DOFA / CAAG track | `gate1_refine_lr2x_seed234` | 2026-11-02T22:50+08:00 | KILL/weak |
| 89.127 | 81.131 | 88.710 | DOFA / CAAG track | `caag_zeroinit_pangaea_seed123` | 2026-11-02T14:11+08:00 | KILL/weak |
| 89.103 | 81.063 | 88.550 | DOFA / CAAG track | `caag_full_pangaea_fttrue_seed42` | 2026-11-02T14:47+08:00 | KILL/weak |
| 89.057 | 80.978 | 88.543 | DOFA / CAAG track | `gate1_s2_segdefault_seed42` | 2026-11-02T13:29+08:00 | KILL/weak |
| 89.046 | 80.980 | 88.780 | DOFA / CAAG track | `caag_availmix_f85o10h05_fttrue_seed234` | 2026-11-02T18:09+08:00 | KILL/weak |
| 89.023 | 80.996 | 88.479 | DOFA / CAAG track | `flora_lite_seed234` | 2026-11-02T23:15+08:00 | KILL/weak |
| 89.019 | 80.904 | 88.616 | DOFA / CAAG track | `caag_zeroinit_pangaea_seed42` | 2026-11-02T14:06+08:00 | KILL/weak |
| 88.978 | 80.864 | 88.775 | DOFA / CAAG track | `caag_availmix_f85o10h05_fttrue_seed42` | 2026-11-02T18:16+08:00 | KILL/weak |
| 88.908 | 80.706 | 88.910 | DOFA / CAAG track | `gate1_refine_ep120_seed234` | 2026-11-02T22:59+08:00 | KILL/weak |
| 88.883 | 80.704 | 88.700 | DOFA / CAAG track | `caag_availmix_f80o15h05_fttrue_seed123` | 2026-11-02T21:45+08:00 | KILL/weak |
| 88.874 | 80.700 | 88.741 | DOFA / CAAG track | `caag_availmix_f70o20h10_cloudlib_fttrue_seed42` | 2026-11-02T20:23+08:00 | KILL/weak |
| 88.791 | 80.555 | 88.754 | DOFA / CAAG track | `caag_availmix_f85o10h05_cloudlib_fttrue_seed234` | 2026-11-02T19:20+08:00 | KILL/weak |
| 88.768 | 80.507 | 88.797 | DOFA / CAAG track | `caag_availmix_f85o10h05_fttrue_seed123` | 2026-11-02T18:23+08:00 | KILL/weak |
| 88.768 | 80.507 | 88.797 | Other | `watch_availmix4_234_then_42_123` | 2026-11-02T18:23+08:00 | KILL/weak |
| 88.735 | 80.447 | 88.707 | DOFA / CAAG track | `caag_availmix_f85o10h05_cloudlib_fttrue_seed123` | 2026-11-02T19:34+08:00 | KILL/weak |
| 88.735 | 80.447 | 88.707 | Other | `watch_availmix_cloudlib_234_then_42_123` | 2026-11-02T19:34+08:00 | KILL/weak |
| 88.702 | 80.390 | 88.707 | DOFA / CAAG track | `caag_availmix_f80o15h05_fttrue_seed234` | 2026-11-02T21:22+08:00 | KILL/weak |
| 88.665 | 80.348 | 88.621 | DOFA / CAAG track | `caag_availmix_f70o20h10_cloudlib_fttrue_seed234` | 2026-11-02T20:16+08:00 | KILL/weak |
| 88.664 | 80.322 | 88.514 | DOFA / CAAG track | `caag_availmix_f85o10h05_cloudlib_fttrue_seed42` | 2026-11-02T19:27+08:00 | KILL/weak |
| … | | | | (24 more in JSON) | | |

## 3. Artifact ensembles / locks / TTA

| name | kind | baseline/test | TTA | notes |
|---|---|---:|---:|---|
| `ensemble_diverse_best3` | ensemble | 91.925 | — | n=3 |
| `ensemble_final_top` | ensemble | 91.925 | — | n=3 |
| `ensemble_never_top3` | ensemble | 91.925 | — | n=3 |
| `tta_unet_d4_160ep_seed234_hv` | tta | 91.759 | 91.826 |  |
| `tta_unet_d4_160ep_seed234_hv_rerun` | tta | 91.759 | 91.826 |  |
| `tta_unet_d4_cedice_seed234_hv` | tta | 91.735 | 91.815 |  |
| `tta_unet_s2_seed42_hv` | tta | 91.717 | 91.822 |  |
| `tta_unet_s2_seed42_hvms` | tta | 91.717 | 91.842 |  |
| `tta_unet_d4_160ep_seed42_hv` | tta | 91.685 | 91.782 |  |
| `ensemble_unet_s2_3seed` | ensemble | 91.606 | — | n=3 |
| `tta_unet_s2_seed123_hv` | tta | 91.544 | 91.558 |  |
| `tta_unet_s2_seed123_hvms` | tta | 91.544 | 91.556 |  |
| `unet_d4_160ep_3seed_lock` | lock | 91.401 | 91.474 | unet_encoder + seg_unet + seg_d4 + 160ep + CE + AdamW 1e-4, S2-only, bs8 |
| `tta_unet_s2_seed234_hv` | tta | 90.943 | 91.004 |  |
| `tta_unet_s2_seed234_hvms` | tta | 90.943 | 91.013 |  |
| `ensemble_unet_d4_160_3seed` | ensemble | 90.864 | — | n=3 |
| `ensemble_unet_d4_160ep_3seed` | ensemble | 90.864 | — | n=3 |
| `tta_unet_d4_160ep_seed123_hv` | tta | 90.759 | 90.813 |  |
| `spire_unet234_verdict` | verdict | 90.516 | — | KILL |
| `scrs_ms_tta_seed42` | tta | 89.568 | 89.864 |  |
| `scrs_tta_seed42` | tta | 89.568 | 89.769 |  |
| `scrs_ms_tta_seed234` | tta | 89.484 | 89.961 |  |
| `scrs_tta_seed234` | tta | 89.484 | 89.773 |  |
| `tta_hydrohard_seed234_hv` | tta | 89.421 | 89.571 |  |
| `scrs_ms_tta_seed123` | tta | 89.286 | 89.724 |  |
| `scrs_tta_seed123` | tta | 89.286 | 89.537 |  |
| `gate1_tta_seed234` | tta | 89.286 | 89.599 |  |
| `gate1_tta_seed123` | tta | 89.190 | 89.497 |  |
| `gate1_tta_seed42` | tta | 89.057 | 89.369 |  |

## 4. In progress

| Exp | last epoch | mtime |
|---|---:|---|
| `caag_availmix_f80o15h05_cloudlib_fttrue_seed42` | 64 | 2026-11-02T22:29+08:00 |
| `caag_availmix_f80o15h05_fttrue_seed123.FAIL_diskfull_ep60` | 60 | 2026-11-02T21:35+08:00 |
| `gate1_mm_segdefault_lr3e4_bs8` | 20 | 2026-11-02T13:19+08:00 |
| `post_cloudlib123_trim_score` | 79 | 2026-11-02T19:34+08:00 |
| `spire_effb3_seed234` | 55 | 2026-11-04T22:44+08:00 |
| `unet_s2_cedice_seed234` | 96 | 2026-11-05T12:00+08:00 |

---

## 5. Next actions

1. Do not revive CAAG-mix / SPIRe / HydroEdge as mainline.
2. Watch `queue_v2.sh`; refresh ensemble after new seeds.
3. Aim ≥92.5 via diverse ensemble → KD, or data-side hard mining.
4. Keep `/hy-tmp` free ≥8GB before new runs.

## Changelog

- 2026-11-05 12:12 Asia/Shanghai: ledger regenerated (104 test rows).
