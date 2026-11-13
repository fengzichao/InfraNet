# InfraNet: Quality-Aware RGB Guidance for Efficient Infrared Object Detection

<p align="center">
  <a href="https://arxiv.org/abs/2607.03795"><strong>Paper (ECCV 2026)</strong></a>
  &nbsp;|&nbsp;
  <a href="https://arxiv.org/html/2607.03795v1">Paper HTML</a>
</p>

> **Release scope.** This repository provides the **InfraNet-IR** implementation with the **YOLOv8-labelled (C2f-based) architecture**, including paired RGB–IR training and IR-only deployment. InfraNet-RGB-IR and other backbone variants reported in the paper are **not included in this code release**. The method and experimental descriptions below follow the **main paper and its supplementary material**. Reported benchmark numbers are taken from those documents, not measured by this repository's smoke tests.

## Overview

InfraNet is an IR-centric object detection framework that uses RGB features as quality-aware guidance. During InfraNet-IR training, a main IR pathway learns detection features, while an auxiliary RGB pathway interacts with it through **three QualGate modules at the P3/P4/P5 neck fusion sites**. At inference, the RGB branch, auxiliary head and QualGate fusion operators are removed. The deployed detector requires **IR input only** (main paper, Secs. 3–4; supplementary, Sec. 2.1).

![InfraNet-IR architecture (main paper, Fig. 2)](https://arxiv.org/html/2607.03795v1/architecture.png)

*Figure: InfraNet-IR dual-stream training and IR-only inference, reproduced from the main paper.*

### QualGate

![QualGate (main paper, Fig. 3)](https://arxiv.org/html/2607.03795v1/fig3.png)

*Figure: QualGate suppression and compensation paths, reproduced from the main paper.*

For an auxiliary RGB-path or previously fused feature $A_{\mathrm{in}}$ and IR features $\{C_k\}$, the main paper defines the task-oriented control score and IR compensation as

$$
q = \sigma\!\left(\mathrm{MLP}_{\mathrm{qual}}\left(\mathrm{GAP}(A_{\mathrm{in}})\right)\right),
\qquad
\alpha(q) = \mathrm{clip}(1.5-q,\,1.0,\,1.5).
$$

The auxiliary contribution is weighted by $q$ and the aligned IR features are gated and amplified; see Eqs. (1)–(4) of the main paper for the full definition. **$q$ is a task-oriented learned control signal, not a separately supervised perceptual image-quality score.**

### Model configuration

| Phase | File | Inputs | Prediction head |
|---|---|---|---|
| Training | [`models/detect/yolov8-c.yaml`](models/detect/yolov8-c.yaml) | Paired **IR then RGB** (6 channels) | Main IR head + auxiliary head (`DualDDetect`) |
| Conversion | `convert.py` | Training checkpoint | Retain IR main pathway and head |
| Inference | [`models/detect/gelanv8-c.yaml`](models/detect/gelanv8-c.yaml) | **IR only** (3 channels) | `DDetect` |

The historical YAML filename `yolov8-c.yaml` denotes the **YOLOv8-labelled backbone setting** used in the paper's comparisons. The implementation itself uses C2f together with YOLOv9-style components such as `SPPELAN` and `RepNCSPELAN4`; it is **not an unchanged, official Ultralytics YOLOv8 detector** (main paper, Sec. 3.2).

## Experimental protocol

The following settings are given in the **main paper, Sec. 5.1**, and the **supplementary material, Sec. 1**:

| Setting | Value |
|---|---|
| Hardware | Single NVIDIA RTX 4090 GPU; PyTorch |
| Epochs | **300** |
| Optimizer | SGD |
| Learning rate | $0.01 \rightarrow 0.0001$ (cosine decay) |
| Warm-up | **3 epochs** |
| Momentum | 0.937 |
| Weight decay | $5\times 10^{-4}$ |
| Batch size | 32 |
| Image size | $640\times640$ |
| Input normalization | $[0,1]$ |
| Auxiliary detection loss weight | $w_{\mathrm{aux}}=0.25$ |
| Augmentation | Mosaic, MixUp, HSV jitter, random horizontal flip, random scaling |
| Paired data | All geometric transformations synchronized between RGB and IR; HSV applied only to RGB |
| Evaluation | Best-validation-mAP checkpoint; single-model, single-scale inference; no test-time augmentation |

The dual-head detection loss is

$$
\mathcal{L} = \mathcal{L}^{(\mathrm{main})}_{\mathrm{det}}
+ 0.25\,\mathcal{L}^{(\mathrm{aux})}_{\mathrm{det}}.
$$

As described in the supplementary material, the two input modalities are spatially aligned and share geometric augmentation parameters. RGB-only HSV changes do not alter geometric alignment. The paper does not specify a universal numeric confidence/NMS threshold for every benchmark; evaluation thresholds should be kept consistent with the intended benchmark protocol.

## Results reported in the paper

The following four rows are the **InfraNet-IR, YOLOv8-labelled** results from the main paper (Tables 1–3) and supplementary material (Table 1 for M³FD). **These are reported paper results, not newly reproduced metrics from the supplied source package.**

| Dataset | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|---|---:|---:|
| LLVIP | **97.3** | **68.8** |
| FLIR-Aligned | **88.5** | **54.0** |
| M³FD | **86.2** | **59.0** |
| DroneVehicle | **84.0** | **63.4** |

For the FLIR-Aligned setting, the main paper reports **28.2M parameters** and **12.5 ms** inference time for this InfraNet-IR model (main paper, Table 2). These figures are reproduced here as **paper-reported measurements**, not as hardware measurements collected by the repository tests.

The full paper also reports independently trained InfraNet-RGB-IR and additional backbone variants. Their published results should be consulted in the paper and supplementary tables; their code is **outside this release's scope**.

## Installation

The repository includes `environment.yml` and `requirements.txt`. They are **code-environment specifications**, not claims about the exact CUDA/Python versions used for the paper experiments. Install a compatible PyTorch/torchvision build for your system first if necessary.

```bash
conda env create -f environment.yml
conda activate infranet-repro
```

## Datasets

The paper evaluates **LLVIP** (15,488 aligned image pairs), **FLIR-Aligned** (5,142), **M³FD** (4,200) and **DroneVehicle** (28,439); see the main paper, Sec. 5.1, for their definitions. Dataset sources:

- [LLVIP](https://bupt-ai-cz.github.io/LLVIP/)
- [FLIR-Aligned](https://github.com/CalayZhou/Multispectral-Pedestrian-Detection-Resource/issues/6)
- [M³FD](https://github.com/JinyuanLiu-CV/TarDAL)
- [DroneVehicle](https://github.com/VisDrone/DroneVehicle)

This source release provides an **LLVIP data-configuration example only**. The example paths below are local directory conventions rather than official benchmark split manifests. Prepare aligned RGB/IR pairs, the corresponding YOLO detection labels, and the desired train/validation partition:

```text
datasets/llvip/
  ir/train/images/     ir/train/labels/
  ir/val/images/       ir/val/labels/
  rgb/train/images/    rgb/train/labels/
  rgb/val/images/      rgb/val/labels/
```

The C training loader expects **IR first, RGB second**. The inherited dataset YAML keys `train_rgb` and `val_rgb` name the second (RGB) stream in this C-only implementation. Data pairs must have matching image identities and spatial alignment.

```bash
python tools/check_dataset_pairs.py --data data/llvip.yaml
```

Dataset releases, annotation mappings and partition files must be checked against the benchmark's evaluation protocol before comparing mAP. This repository does not contain the original benchmark datasets or every paper checkpoint.

## Training

Train **InfraNet-IR** using the paper's 300-epoch protocol (see the training settings table above):

```bash
python train_dual.py \
  --cfg models/detect/yolov8-c.yaml \
  --data data/llvip.yaml \
  --hyp data/hyps/hyp.scratch-high.yaml \
  --epochs 300 --batch-size 32 --img 640 \
  --optimizer SGD --cos-lr --device 0 --weights '' \
  --name infranet_ir
```

The six-channel training model uses two detection heads, with an auxiliary loss weight of $0.25$. Its paired validation routine reads both modalities during training; **the released inference graph uses only IR**.

## Convert and evaluate the IR-only model

After training, convert the checkpoint to the IR-only deployment architecture:

```bash
python convert.py \
  --source runs/train/infranet_ir/weights/best.pt \
  --target-cfg models/detect/gelanv8-c.yaml \
  --output weights/infranet_ir.pt
```

The evaluation and detection scripts use the **converted 3-channel IR-only** checkpoint:

```bash
python val.py \
  --weights weights/infranet_ir.pt \
  --data data/llvip.yaml --img 640 --batch-size 16 --device 0

python detect.py \
  --weights weights/infranet_ir.pt \
  --source path/to/ir/images --data data/llvip.yaml --img 640
```

**Do not use the six-channel training checkpoint for IR-only evaluation.** Do not use the converted three-channel checkpoint with `val_paired.py`.

## Code tests and scope

The following local tests exercise module behavior, dataset pairing, the training-to-inference weight conversion and IR-only inference with synthetic inputs:

```bash
python tools/check_release.py
python tests/test_modality_contract.py
python tests/smoke_qualgate.py
python tests/test_actual_gate_routing.py
python tests/test_conversion_scaled.py
python tests/test_convert_cli_scaled.py
python tests/test_training_loss.py
python tests/test_dataset_pairing.py
python tests/test_paired_augmentation.py
python tests/test_c_paired_validation.py
python tests/test_ir_eval_scaled.py
python tests/test_training_smoke.py
```

The final smoke test runs a complete one-epoch scaled C training, checkpoint conversion, and IR-only evaluation on synthetic paired images. These are **engineering tests**, not four-dataset mAP reproductions. The publicly released C-only code should not be interpreted as an implementation of the independent InfraNet-RGB-IR architecture or of all backbone settings.

## License and citation

This implementation builds on [YOLOv9](https://github.com/WongKinYiu/yolov9) and preserves the inherited **GPL-3.0** licensing conditions (see [`LICENSE`](LICENSE)). Refer to the original repositories and datasets for their respective licenses.

```bibtex
@misc{feng2026infranet,
  title={InfraNet: Quality-Aware RGB Guidance for Efficient Infrared Object Detection},
  author={Feng, Zichao and Zhu, Haodong and Yang, Jingying and Xu, Sheng and Ren, Yangyang and Yang, Yuguang and Liu, Xuhui and Zhang, Juan and Wang, Tian and Yang, Linlin and Zhang, Baochang},
  year={2026},
  eprint={2607.03795},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
