## Environment

```bash
conda create -n muon python==3.9.19
conda activate muon
pip install -r requirements.txt

FT：

```bash
MODEL=Qwen/Qwen2.5-1.5B TASK=SST2 TRAIN=1000 GPU=0 USE_MUON=false bash scripts/finetune.sh
```

FT-Muon:

```bash
MODEL=Qwen/Qwen2.5-1.5B TASK=SST2 TRAIN=1000 GPU=0 USE_MUON=true DRIFT_AWARE=false bash scripts/finetune.sh
```

FT-Muon-DA:

```bash
MODEL=Qwen/Qwen2.5-1.5B TASK=SST2 TRAIN=1000 GPU=0 USE_MUON=true DRIFT_AWARE=true bash scripts/finetune.sh
```

GS_Muon:

```bash
MODEL=Qwen/Qwen2.5-1.5B TASK=SST2 TRAIN=1000 GPU=0 bash scripts/GS_Muon.sh
```

Ablation study

```bash
# hard reset only
--da_momentum false --da_hard_reset true

# GG-NS with no gradient steering
--da_momentum false --gg_ns_eta_g 0.0

# Full DA-Muon with GG-NS
--da_momentum true --gg_ns_eta_g 0.25
```
