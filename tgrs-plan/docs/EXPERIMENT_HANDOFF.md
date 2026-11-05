# TGRS / Sen1Floods11 Experiment Ledger

**Updated:** 2026-11-05 12:12 Asia/Shanghai  
**Machine:** 恒源云 i-2 (`ssh -p 37782 root@i-2.gpushare.com`) · 工作区 `/hy-tmp/tgrs-plan`  
**Task:** 语义分割 · **Sen1Floods11** · 默认 **S2-only** · 主指标 **test mIoU**（附 Water IoU）

> 下次接手前必读：§1 结论 → §2 排行榜 → §6 下一步。  
> 机器上自动表：`/hy-tmp/tgrs-plan/docs/EXPERIMENT_LEDGER.md`  
> 刷新：`python3.11 /hy-tmp/tgrs-plan/scripts/update_experiment_ledger.py`  
> 本地副本：`/workspace/docs/EXPERIMENT_LEDGER.md` · JSON：`tgrs-plan/artifacts/experiment_ledger.json`

---

## 0. 当前状态（2026-11-05 12:12 Asia/Shanghai）

| 项 | 状态 |
|---|---|
| 实例 | **无卡启动**（GPU 到期后重启）· `nvidia-smi` → No devices |
| SSH | **已连通** `root@i-2.gpushare.com:37782` |
| 磁盘 | `/hy-tmp` **13G/50G 已用，约 38G 空闲（25%）** |
| 队列/训练 | **全部停止**（无 GPU） |
| 中断任务 | `unet_s2_cedice` seed234 · **停在 epoch 96/160** · 有 `checkpoint__best.pth`（ep90 保存）· **尚无 test mIoU** |
| 最优（未变） | ensemble **91.925** · 单模 **91.759** |

**有卡后第一件事：** 从 `checkpoint__best.pth` 续训/评测 `unet_s2_cedice`，再跑 `queue_v2` 剩余 seed999 / 多样性 seed，然后刷新 ledger。

---

## 1. 一句话结论

- **Clean SOTA 主战场**已被强 **UNet** 占住（单模 ~91.4–91.8）；公开榜 UNet ≈ **91.42**。
- **DOFA Gate1** 锁定 **89.178**；**SCRS** 相对 Gate1 **+0.27**（可写小创新，远不够刷榜）。
- **CAAG / mix / CAFE / FLoRA / SPIRe / HydroEdge / HydroHard / GGU** 均未稳定超过强 UNet → 主表勿吹，作 ablation/附录。
- **当前最优：** 多样 ensemble **91.925**；最优单模 **UNet-d4 160ep seed234 = 91.759**（HV-TTA **91.826**）；最强单模+TTA **91.842**（S2 seed42 HVMS）。
- **目标未达成：** 单模或 ensemble **≥92.5** 的大涨点还没有。

---

## 2. 排行榜（锁定参考）

| Rank | Setting | test mIoU | Water | Notes |
|---:|---|---:|---:|---|
| 1 | Diverse ensemble (d4@234 + s2@42 + cedice@234) | **91.925** | 85.93 | `artifacts/ensemble_diverse_best3.json` |
| 2 | UNet-S2 seed42 + HVMS-TTA | **91.842** | 85.80 | 最强单模+TTA |
| 3 | UNet-d4 160ep seed234 + HV-TTA | **91.826** | 85.75 | `tta_best_rerun` 复核同值 |
| 4 | **UNet-d4 160ep seed234** | **91.759** | 85.64 | **最强单模（无 TTA）** |
| 5 | UNet-d4 CE+Dice seed234 | 91.735 | 85.61 | ≈持平 |
| 6 | UNet-S2 seed42 | 91.717 | 85.58 | |
| 7 | UNet-S2 3-seed ensemble | 91.606 | 85.38 | |
| 8 | UNet-d4 160ep 3-seed mean | 91.401 | — | lock；HV-TTA mean 91.474 |
| — | Public UNet（外榜近似） | ~91.42 | — | SOTA2 Sen1Floods11 (test) |
| — | SCRS + MS-TTA | ~89.85 | — | DOFA 线峰值 |
| — | SCRS 3-seed | 89.446±0.12 | 81.66 | +0.268 vs Gate1 |
| — | Gate1 DOFA 3-seed | 89.178±0.12 | 81.20 | **协议锁** |

### 判定门槛

| Gate | 规则 |
|---|---|
| KILL | test mIoU **&lt; 91.50**（相对 UNet-d4 基线） |
| competitive | 91.50–91.75 |
| PROMOTE | 单模 **≥ 92.0** |
| TARGET | **≥ 92.5**（单模或 ensemble） |

---

## 3. 按路线汇总

### 3.1 UNet family（主涨点来源）

| Exp | test | Water | best val | Verdict |
|---|---:|---:|---:|---|
| unet_d4_160ep_seed234 | **91.759** | 85.638 | 91.737 | **锁定单模 SOTA** |
| unet_d4_cedice_seed234 | 91.735 | 85.611 | 91.762 | NEAR |
| unet_s2_seed42 | 91.717 | 85.583 | 91.099 | NEAR |
| unet_d4_160ep_seed42 | 91.685 | 85.523 | 91.754 | |
| unet_s2_seed123 | 91.544 | 85.292 | 91.222 | |
| unet_d4_seed123 (80ep级) | 91.527 | 85.246 | 91.186 | |
| unet_d4_lovasz_seed234 | 91.475 | 85.168 | 91.403 | |
| unet_mm_d4_seed123 | 91.438 | 85.109 | 91.206 | MM 未超 S2 强模 |
| unet_mm_d4_seed234 | 91.399 | 85.026 | 91.034 | |
| r34_d4_seed234 | 91.269 | 84.831 | 90.670 | |
| unet_d4_seed234 | 91.247 | 84.776 | 90.972 | 短训弱于 160ep |
| unet_s2_seed234 | 90.943 | 84.198 | 90.833 | |
| unet_d4_200ep_seed234 | 90.905 | 84.145 | 91.934 | **更长更差（过拟合）** |
| unet_recipe*_ / 240ep | 90.6–91.0 | — | 高 val | 复杂 recipe 未赢 |
| **unet_s2_cedice_seed234** | **pending** | — | val 峰值见 log | **中断 @ep96/160**；best ckpt@ep90 |

**读数：** `encoder=unet_encoder` + `seg_unet` + **`seg_d4`** + **160ep** + **CE** + AdamW 1e-4 最稳。CE+Dice ≈持平。200ep / 重 recipe 易 val 虚高、test 掉点。

### 3.2 SCRS（可写小创新）

| Exp | test | Notes |
|---|---:|---|
| SCRS 3-seed | **89.446** | +0.268 vs Gate1；`encoder=scrs_dofa` |
| SCRS + flip-TTA | 89.693 | |
| SCRS + MS-TTA | ~89.85 | DOFA 线最好 |
| SCRS partial/full FT | ≤89.49 | **KILL** |

### 3.3 Gate1 / CAAG / mix

| Exp | test | Verdict |
|---|---:|---|
| Gate1 S2 UPerNet 3-seed | **89.178±0.115** | **LOCKED 协议** |
| CAAG full FIXED 3-seed | ~89.21 | ≈噪声，非涨点 |
| CAAG avail/cloud mix | 多为 87.x–88.x | **伤害 clean** → 附录 |
| CAFE / FLoRA | ~89.0 | **KILL** |
| 早期错误配置 Gate1 | 78–85 | 已 supersede |

### 3.4 大胆创新臂（全部未超强 UNet）

| Exp | 思路 | test | Verdict |
|---|---|---:|---|
| SPIRe-UNet | SCRS+指数+CE+Dice | **90.516** | **KILL**（val 92.01 过拟合） |
| HydroHard | 边界加权 FT | ~90.47 / TTA ~90.60 | **KILL** |
| HydroEdge | Laplacian 边缘 + NDWI | 89.964 | **KILL** |
| HydroEdge 200ep | | 90.369 | **KILL** |
| HydroEdge+CE+Dice | | 90.217 | **KILL** |
| HydroWide | 更宽 UNet+edge | 89.472 | **KILL** |
| GGU base/d4/idx | DOFA 门控注入 UNet | 90.87–91.12 | &lt; plain UNet |
| SPIRe-EffB3 | | 未完成 | HF/预训练不稳，跳过 |

**可视化诊断：** 大水体已饱和；短板是**细河 FN**与**云 FP**。再堆光谱/边缘 stem 未能泛化到 test。

### 3.5 Ensemble / TTA（摘要）

| Exp | mIoU |
|---|---:|
| ensemble_diverse_best3 | **91.925** |
| ensemble_unet_s2_3seed | 91.606 |
| ensemble_unet_d4_160ep_3seed | 90.864 |
| unet_d4_160ep_3seed_lock (mean) | 91.401 / HV 91.474 |

机器扫描共 **104** 条含 test 的 log（2026-11-05 12:10 刷新）。

---

## 4. 中断 / 未完成清单（有卡后续）

| Exp | 状态 | 备注 |
|---|---|---|
| `unet_s2_cedice_seed234` | **中断 ep96/160** | best@ep90；`queue_v2` 用 `grep '[test] Mean'` 判完成——恢复后会**整段重跑** unless 改 resume |
| `unet_d4_160ep_seed999` | 未开训 | `queue_v2` 下一步 |
| `unet_s2_160` seed999 / seeds 7,77,1234 | 排队 | 多样性 ensemble |
| KD（diverse → student） | 未做 | ≥92 主候选之一 |
| 数据侧 hard mining / water copy-paste | 未做 | 打细河 FN、云 FP |

---

## 5. 关键路径 & 清盘策略

```
代码/计划:  /hy-tmp/tgrs-plan
数据:      /hy-tmp/tgrs-plan/data/sen1floods11
Runs:      /hy-tmp/tgrs-plan/runs/
Logs:      /hy-tmp/logs/
Artifacts: /hy-tmp/tgrs-plan/artifacts/
Ledger:    /hy-tmp/tgrs-plan/docs/EXPERIMENT_LEDGER.md
Handoff:   /hy-tmp/tgrs-plan/docs/EXPERIMENT_HANDOFF.md   # 本叙事副本
刷新脚本:  /hy-tmp/tgrs-plan/scripts/update_experiment_ledger.py
队列:      /hy-tmp/tgrs-plan/scripts/queue_v2.sh
```

**清盘后保留的强 ckpt：**
- `pangaea_unet_d4_160ep_seed{234,42,123}`
- `pangaea_unet_s2_seed{234,42,123}`
- `pangaea_unet_d4_cedice_seed234`
- `pangaea_unet_s2_cedice_seed234`（中断中，先留 best）
- Gate1 / SCRS DOFA runs（对照）

**红线：** `/hy-tmp` 使用 &gt;85% 时先删 `checkpoint_*_ep*.pth` 与失败创新 run，再开训。曾因磁盘 100% 导致 GPU 空转。

---

## 6. 下一步（接续清单）

1. **挂卡后**确认 `nvidia-smi`，再决定：续训 `unet_s2_cedice` 或让 `queue_v2` 重跑该 job。
2. **不要**再主攻 CAAG-mix / SPIRe / HydroEdge 同类先验堆叠（已证伪）。
3. 冲 ≥92.5 的优先方向：
   - 异架构/异预处理 **多样 ensemble → KD 压成单模**
   - **数据侧**（water copy-paste / hard mining）打细河 FN、云 FP
   - 级联 refine / 更强 decoder（先看磁盘）
4. 每次实验结束（`queue_v2` 已挂钩）：
   ```bash
   python3.11 /hy-tmp/tgrs-plan/scripts/update_experiment_ledger.py
   ```
5. 写论文时：主表以 UNet clean SOTA + SCRS（相对 Gate1）为故事；大创新需等真正 ≥92 的方法。

---

## 7. Changelog

| 时间 (Asia/Shanghai) | 内容 |
|---|---|
| 2026-11-05 12:05 | 初版 ledger：汇总 Gate1→SCRS→UNet→SPIRe/Hydro*→ensemble |
| 2026-11-05 12:10 | SSH 恢复（无卡）；部署 `update_experiment_ledger.py`；机器扫描 **104** 条 test；ensemble 仍 **91.925** |
| 2026-11-05 12:12 | 记录实例到期→无卡启动；`unet_s2_cedice` 中断@ep96；`queue_v2` 挂 ledger 刷新钩子 |
| （每次跑 updater） | 自动重扫 logs/artifacts 更新机器上表格 |

---

## 8. 快速命令

```bash
# 登录
ssh -p 37782 root@i-2.gpushare.com

# 进度 / 有无卡
nvidia-smi
df -h /hy-tmp
tail -30 /hy-tmp/logs/queue_v2.outer
python3.11 /hy-tmp/tgrs-plan/scripts/update_experiment_ledger.py
less /hy-tmp/tgrs-plan/docs/EXPERIMENT_LEDGER.md
less /hy-tmp/tgrs-plan/docs/EXPERIMENT_HANDOFF.md

# 最强单模 / ensemble / 中断 run
ls /hy-tmp/tgrs-plan/runs/pangaea_unet_d4_160ep_seed234/*/checkpoint__best.pth
ls /hy-tmp/tgrs-plan/runs/pangaea_unet_s2_cedice_seed234/*/checkpoint__best.pth
cat /hy-tmp/tgrs-plan/artifacts/ensemble_diverse_best3.json
```
