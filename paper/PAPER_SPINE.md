# Paper Spine — TipGuided UniMTS

**Skills used:** PaperSpine (local evidence mode), ARS academic-paper plan, Research-Paper-Writing, Nature-writing (algorithmic/methods framing), ARIS claims-drafting.

**Venue target:** ML conference (NeurIPS / ICML / ICLR style) or ubiquitous computing (IMWUT) — draft uses NeurIPS-like structure.

**Research mode:** Public literature + private experimental logs (`/hy-tmp/logs`, UniMTS official splits). No invented metrics.

## Core question
How should a motion time-series foundation model (UniMTS) be adapted under **cross-dataset few-shot** HAR without overfitting the support set?

## One-sentence claim
Encoder fine-tuning guided by a Tip-cache prior (text + support affinity) outperforms both frozen Tip-Adapter and plain Full Fine-Tuning on UniMTS’s 18-dataset few-shot protocol.

## Evidence chain
1. **Problem evidence:** Zero-shot prompt/location hacks fail; frozen Tip+ saturates (~+0.5%).
2. **Method:** TipGuided — freeze text tower; update ST-GCN + Tip/CE; logits \(z=z_t+\alpha AV+\gamma Wf(x)\).
3. **Main result:** TipGuided > Full FT for all \(k\in\{1,2,3,5,10\}\).
4. **Ablations:** Tip-F / Tip+ / probe / LightFT / anneal; fuse helps low-\(k\), CE-primary mid/high-\(k\).

## Contribution list (final)
C1. Diagnose cross-dataset few-shot as the operative UniMTS transfer problem; show frozen Tip saturates.
C2. Propose TipGuided adaptation coupling Tip cache priors with encoder FT.
C3. Establish 18-dataset gains over Full FT under official `few_shot_data_2` indices + LOO \((\beta,\alpha)\).

## Non-claims
- Inventing Tip-Adapter; prompt engineering as a method; LOO as novelty.
