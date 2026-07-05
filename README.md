# SAMamba3D

**Adapting Segment Anything for generalizable three-dimensional segmentation of multiphase pore-scale images**


SAMamba3D is a parameter-efficient 3D segmentation framework for multiphase pore-scale X-ray images. It adapts the pretrained **Segment Anything Model (SAM)** image encoder to volumetric data by coupling it with a multi-scale **Mamba** 3D encoder, progressive cross-scale feature interaction, and a hybrid 3D decoder.

The model is designed for supervised semantic segmentation of volumetric rock images, for example oil/brine/rock segmentation in pore-scale images. Unlike the original promptable 2D SAM interface, this repository focuses on **automatic 3D voxel-wise segmentation** from volumetric inputs.

> Paper: Zhang, R., Song, X., Zhu, L., Bijeljic, B., Li, G., & Blunt, M. J. (2026). *SAMamba3D: Adapting segment anything for generalizable three-dimensional segmentation of multiphase pore-scale images*. **Advances in Geo-Energy Research, 21(2)**. https://doi.org/10.46690/ager.2026.08.03


## Highlights

- **SAM-to-3D adaptation**: converts SAM-style ViT image encoding to volumetric encoding with 3D patch embedding and 3D window attention.
- **Mamba volumetric context modeling**: uses a four-scale 3D Mamba encoder for efficient long-range context in volumes.
- **Progressive co-encoding**: injects Mamba features into SAM tokens through shallow cross-scale adapters and deeper bidirectional bridges.
- **Parameter-efficient fine-tuning**: keeps most SAM block parameters frozen in early training and progressively enables LayerNorm, reverse bridges, and LoRA bypasses.
- **Hybrid 3D decoder**: fuses SAM neck features, Mamba multi-scale outputs, intermediate co-encoded features, FiLM modulation, and high-resolution skips.
- **Pore-scale segmentation losses**: combines soft Dice, Tversky, focal, and interior-aware losses to improve phase boundaries and small structures.
- **Sliding-window inference**: supports large volumes via patch-wise inference with optional Gaussian weighting.

---

## Repository structure

```text
SAMamba-3D/
├── AdapterSAM/
│   └── image_encoder_3d.py        # 3D SAM-style ViT encoder, 3D adapters, 3D window attention
├── Combined_dataloader.py         # Numpy volume loading, patch sampling, augmentation, normalization
├── Compoundloss.py                # Focal, boundary, Tversky, enhanced, and RockCore losses
├── Config.py                      # Default patch, training, path, and seed configuration
├── SAM_train.py                   # Main training entry point
├── SAMamba3D.py                   # SAM-Mamba co-encoding model and hybrid decoder
├── early_stopping.py              # Early stopping utility
├── image_slice_view.py            # 2D slice and 3D overview visualization helpers
├── mamba_encoder.py               # 3D Mamba encoder
├── memory_cal.py                  # GPU/CPU memory measurement utilities
├── model_inference.py             # Sliding-window inference and evaluation script
├── requirements.txt               # Python dependencies captured for the current environment
├── LICENSE                        # Apache-2.0 license
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ImperialCollegeLondon/SAMamba-3D.git
cd SAMamba-3D
```

### 2. Create an environment

A CUDA-enabled Linux environment is recommended.

```bash
conda create -n samamba3d python=3.10 -y
conda activate samamba3d
```

### 3. Install PyTorch

The included `requirements.txt` was generated with PyTorch 2.4.1 and CUDA 12 packages. Install the PyTorch build matching your CUDA driver.

Example for CUDA 12.4:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

For other CUDA versions, follow the official PyTorch installation selector.

### 4. Install project dependencies

```bash
pip install -r requirements.txt
```

The code imports `mamba_ssm` in `mamba_encoder.py`. If it is not installed by your environment, install it separately:

```bash
pip install causal-conv1d mamba-ssm
```

If `mamba-ssm` compilation fails, check that your CUDA toolkit, PyTorch CUDA version, compiler, and GPU architecture are compatible.

### 5. Verify the environment

```bash
python - <<'PY'
import torch
import monai
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MONAI:", monai.__version__)
try:
    import mamba_ssm
    print("mamba_ssm: OK")
except Exception as e:
    print("mamba_ssm import failed:", e)
PY
```

---

## SAM checkpoints

SAMamba3D initializes the SAM image encoder from pretrained SAM checkpoints. The recommended default is `vit_b`.
Create a checkpoint directory:

```bash
mkdir -p checkpoints/sam
```

Download the SAM ViT-B checkpoint:

```bash
wget -O checkpoints/sam/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Optional SAM checkpoints:

| Model type | Checkpoint filename | Notes |
|---|---|---|
| `vit_b` | `sam_vit_b_01ec64.pth` | Recommended default for this repository |
| `vit_l` | `sam_vit_l_0b3195.pth` | Architecture branch exists in `SAMamba3D.py` |
| `vit_h` | `sam_vit_h_4b8939.pth` | Parser option exists, but current model construction is not fully specialized for ViT-H |

---

## Data preparation

### Expected input format

Training uses NumPy arrays loaded from `.npy` files:

```text
image volume:  float32 array with shape [D, H, W]
label volume:  int32/int64 array with shape [D, H, W]
```

Inference supports `.npy` and `.tif` volumes in `model_inference.py`, although `.npy` is the most direct path with the current code.

The default label convention in the evaluation helper is:

```text
0 = Unknown / background
1 = Oil
2 = Brine
3 = Rock
```

For other segmentation tasks, update `num_classes`, `label_mapping`, and the class names used in `model_inference.py`.


## Inference

Inference is implemented in `model_inference.py` with sliding-window prediction over a full volume.

### 1. Required edits before running

The current inference script contains project-specific paths and one local dependency that is not included in the repository. Before running, edit the `if __name__ == "__main__":` block and the `test_model()` function.

Set a device explicitly:

```python
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Set model and data paths:

```python
config.data_path = "data/SSa/image.npy"
config.labels_path = "data/SSa/label.npy"  # or None for unlabeled inference
checkpoint_path = "outputs/run_samamba3d_vitb_patch96/checkpoints/best_model.pth"
config.result_dir = "outputs/run_samamba3d_vitb_patch96/results"
```

Set patch and stride:

```python
config.patch_size = (128, 128, 128)
config.stride = tuple(int(p * (1 - 0.7)) for p in config.patch_size)
```

If you do not have the local `data_transfer.py` module, remove or replace the following domain-transfer block inside `test_model()`:

```python
from data_transfer import DomainTransfer
dt = DomainTransfer()
test_data = dt.percentile_normalization(test_data, target_data)
```

A minimal replacement is:

```python
test_data = test_data.astype(np.float32)
```

### 2. Run inference

```bash
python model_inference.py
```

### 3. Inference outputs

The script saves:

- A predicted label volume, for example `1PV_pred.npy`.
- A 2D slice comparison figure.
- A 3D overview prediction figure.
- Optional evaluation metrics when ground-truth labels are available.
- Runtime and memory statistics.

### 4. Prediction behavior

The inference engine:

1. Loads a full 3D volume.
2. Normalizes it using either current volume statistics or saved training statistics from `data_stats.npy`.
3. Splits the volume into overlapping 3D patches.
4. Runs `model.forward_stage1()` on each patch.
5. Applies softmax to class logits.
6. Averages overlapping predictions.
7. Returns `argmax` class labels with shape `[D, H, W]`.

---

## Python API

Minimal model construction:

```python
import torch
from SAMamba3D import SAM_Mamba_3D_CoEncoding

mamba_config = {
    "in_chans": 1,
    "depths": [2, 2, 2, 2],
    "dims": [48, 96, 192, 384],
    "drop_path_rate": 0.1,
    "out_indices": [0, 1, 2, 3],
}

model = SAM_Mamba_3D_CoEncoding(
    model_type="vit_b",
    mamba_config=mamba_config,
    num_classes=4,
    in_chans=1,
    out_chans=256,
    lora_rank=8,
    lora_alpha=16.0,
).cuda()

model.set_training_stage("A", sam_checkpoint="checkpoints/sam/sam_vit_b_01ec64.pth")

x = torch.randn(1, 1, 96, 96, 96).cuda()  # [B, C, D, H, W]
logits = model.forward_stage1(x)           # [B, num_classes, D, H, W]
print(logits.shape)
```

Training-mode forward pass returns additional intermediate outputs:

```python
outputs = model.forward_stage1(x, training=True)
print(outputs.keys())
```

---

## Model architecture

```text
Input volume [B, 1, D, H, W]
        │
        ├── MambaEncoder
        │      ├── scale0: [B,  48, D/2,  H/2,  W/2]
        │      ├── scale1: [B,  96, D/4,  H/4,  W/4]
        │      ├── scale2: [B, 192, D/8,  H/8,  W/8]
        │      └── scale3: [B, 384, D/16, H/16, W/16]
        │
        ├── EarlyFusionStem + 3D PatchEmbed
        │      └── SAM-style 3D tokens
        │
        ├── CoEncodingEncoder
        │      ├── 3D SAM ViT blocks
        │      ├── LoRA bypasses
        │      ├── shallow Mamba → SAM cross-scale adapters
        │      ├── deep bidirectional SAM ↔ Mamba bridges
        │      ├── Mamba global controller
        │      └── DACFM feature fusion nodes
        │
        └── HybridCoDecoder
               ├── SAM neck features
               ├── Mamba skip features
               ├── FiLM modulation
               ├── intermediate co-encoded features
               ├── high-resolution skip
               └── voxel logits [B, num_classes, D, H, W]
```

### Main components

| Component | File | Purpose |
|---|---|---|
| `ImageEncoderViT_3d_v2` | `AdapterSAM/image_encoder_3d.py` | 3D adaptation of SAM ViT blocks with 3D window attention |
| `MambaEncoder` | `mamba_encoder.py` | Multi-scale volumetric context encoder |
| `SAMBlockLoRABypass` | `SAMamba3D.py` | Lightweight LoRA residual path around SAM blocks |
| `CrossScaleAdapter` | `SAMamba3D.py` | Injects aligned Mamba features into SAM token space |
| `BidirectionalBridge` | `SAMamba3D.py` | Enables Mamba → SAM and optional SAM → Mamba interaction |
| `MambaGlobalController` | `SAMamba3D.py` | Produces routing scores, injection strengths, and FiLM parameters |
| `DACFM` | `SAMamba3D.py` | Dynamic attention-based cross-feature fusion module |
| `HybridCoDecoder_v5` | `SAMamba3D.py` | Reconstructs full-resolution segmentation logits |
| `RockCoreLoss` | `Compoundloss.py` | Composite loss for pore-scale phase segmentation |

---

## Loss and metrics

The default trainer uses `RockCoreLoss`, which combines:

- Soft-label Dice loss.
- Tversky loss.
- Interior-weighted focal loss.
- Interior-weighted cross-entropy.

The interior weighting down-weights uncertain boundary margins and emphasizes more reliable interior voxels. This is useful for pore-scale images where human labels can be ambiguous at phase boundaries.

Validation Dice is computed over foreground classes by default, excluding class `0`.

During inference with labels, the script reports per-class:

- Precision
- Recall
- Dice
- IoU

and overall accuracy, Dice, and IoU on labeled regions.

---

## Checkpoints and outputs

Training checkpoints are saved as PyTorch dictionaries with fields such as:

```python
{
    "epoch": epoch,
    "stage": current_stage,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_dice": best_dice,
    "config": config,
    "scaler_state_dict": scaler.state_dict(),  # when AMP is enabled
}
```

Typical output tree:

```text
outputs/
└── run_samamba3d_vitb_patch96/
    ├── checkpoints/
    │   ├── best_model.pth
    │   ├── stage_i_epoch_50.pth
    │   └── stage_i_final.pth
    ├── logs/
    │   └── train.log
    └── results/
        ├── 1PV_pred.npy
        ├── slice_1PV_33_comparison.png
        ├── 3Dmamba_1PV_pred.png
        ├── 3Dmamba_1PV_gt.png
        └── 3Dmamba_1PV_raw.png
```

---

## Current implementation notes

The current repository is research code and contains several project-specific defaults. Check these before public release or before running on a new machine.

1. **Training data lists are empty by default.** Populate `dataset_img_paths`, `dataset_label_paths`, and `data_names` in `SAM_train.py`.
2. **Inference requires `config.device`.** Add `config.device = torch.device(...)` in `model_inference.py` before calling `test_model()`.
3. **`data_transfer.py` is not present in the root repository.** Remove or replace the `DomainTransfer` block in `model_inference.py` unless you add that module.
4. **`stage_ii` is scaffolded but not implemented.** The parser exposes `stage_ii` and `both`, but `trainer.py` currently provides `train_stage_i()` and `fine_tuning()`, not `train_stage_ii()`.
5. **`vit_b` is the safest default.** The code has a `vit_l` branch, while `vit_h` is present as a parser choice but should be treated as experimental unless fully adapted.
6. **`mamba_ssm` must be installed.** It is imported by `mamba_encoder.py`; install it separately if missing from your environment.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'mamba_ssm'`

Install Mamba SSM and its CUDA convolution dependency:

```bash
pip install causal-conv1d mamba-ssm
```

If compilation fails, align PyTorch, CUDA, GCC, and GPU architecture versions.

### `SAM checkpoint not found`

Check that `--sam_checkpoint` points to an existing `.pth` file:

```bash
ls checkpoints/sam/sam_vit_b_01ec64.pth
```

### CUDA out of memory

Use a smaller patch size in `Config.py`:

```python
patch_size = (64, 64, 64)
stride = tuple(int(p * (1 - 0.5)) for p in patch_size)
```
Also reduce `batch_size` to `1`, disable AMP only if it causes numerical issues, and avoid large validation patch counts.

### Label index out of range

Ensure labels are integers in `[0, num_classes - 1]`. If your dataset uses arbitrary values, define `label_mapping` before calling `data_loaders()`.

---

## Citation

If you use this code or model in your research, please cite:

```bibtex
@article{zhang2026samamba3d,
  title   = {SAMamba3D: Adapting segment anything for generalizable three-dimensional segmentation of multiphase pore-scale images},
  author  = {Zhang, Rui and Song, Xianzhi and Zhu, Linqi and Bijeljic, Branko and Li, Gensheng and Blunt, Martin J.},
  journal = {Advances in Geo-Energy Research},
  volume  = {21},
  number  = {2},
  year    = {2026},
  doi     = {10.46690/ager.2026.08.03}
}
```

---

## License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

The SAM pretrained checkpoints are distributed by the Segment Anything project. Check the original SAM repository for its model and dataset license terms.

---

## Acknowledgements

This project builds on ideas and software from:

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [MONAI](https://monai.io/)
- [Mamba SSM](https://github.com/state-spaces/mamba)
- PyTorch, NumPy, SciPy, scikit-image, SimpleITK, and related scientific Python libraries

