# Claim–Evidence Map

| ID | Claim | Evidence | Status |
|----|-------|----------|--------|
| A1 | Cross-dataset few-shot is the right problem (not ZS prompt tuning) | Location/text conditioning dropped ZS 0.444→~0.278 in early runs; protocol uses 18 datasets + few_shot_data_2 | supported (internal log; keep as diagnostic) |
| A2 | Frozen Tip-F helps but saturates | Tip-F vs Tip+ at k=5: 0.705→0.709 (~+0.5%); tip_plus_* logs | supported |
| A3 | TipGuided beats Tip-F | Gaps +4.0% (k=1) to +8.4% (k=2); results_snapshot.json | supported |
| A4 | TipGuided beats Full FT at all k | k=1..10 gaps +0.35%..+3.8%; results_snapshot.json | supported |
| A5 | Low-k prefers Tip+CE fuse; mid/high-k prefers CE-primary | best files: fuse@k1/2/3, ce_longlow@k5, ce@k10 | supported (ablation) |
| A6 | Linear probe is weak under domain shift | probe k=5 0.519 << Tip-F 0.705 | supported |
| L1 | Multi-seed significance at k=5 (+0.35%) | Not yet run | **needs evidence** |
| L2 | Unified tip-weight schedule without per-k variant pick | Not yet run | **needs evidence** |
| L3 | Beats LoRA/ProtoNet under same budget | Not yet run | **needs evidence** |

**Abstract/Intro rule:** Only use A2–A6 as hard claims; mention L1–L3 as future work / limitations.
