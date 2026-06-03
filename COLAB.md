# DiffuSuite on Google Colab

Colab is the recommended place to train the custom CIFAR-10 checkpoints and run
the optional Stable Diffusion integrations. GPU types and availability vary by
session, so begin by checking the assigned accelerator.

For a direct runnable notebook, open:

```text
notebooks/DiffuSuite_Colab.ipynb
```

The notebook assumes your Drive folder is:

```text
/content/drive/MyDrive/diffu_suite/data/cifar10_dataset
```

## 1. Select a GPU Runtime

In Colab, choose **Runtime > Change runtime type > GPU**, then run:

```python
!nvidia-smi
```

## 2. Open the Project

Upload this repository or clone your GitHub copy, then enter its root:

```python
%cd /content/diffu_suite
!pip install -r requirements.txt
```

Place the extracted dataset at:

```text
data/cifar10_dataset/
├── train/{0..9}/*.png
└── test/{0..9}/*.png
```

Audit it:

```python
!python3 scripts/validate_dataset.py --hashes
```

## 3. Train the Custom Checkpoints

Start conservatively on an unfamiliar GPU:

```python
!python3 training/train_ddpm.py \
  --schedule cosine \
  --output-dir runs/cifar10_cosine \
  --batch-size 64 \
  --epochs 100 \
  --workers 2
```

Then train the controlled comparison:

```python
!python3 training/train_ddpm.py \
  --schedule linear \
  --output-dir runs/cifar10_linear \
  --batch-size 64 \
  --epochs 100 \
  --workers 2
```

If memory permits, increase `--batch-size` to `128`. If CUDA runs out of
memory, reduce it to `32`. Training automatically uses CUDA float16 AMP.

## 4. Export Samples and Reverse Videos

```python
!python3 inference/sample_custom.py \
  runs/cifar10_cosine/checkpoints/latest.pt \
  --output artifacts/generated/cifar10_cosine.png \
  --trajectory-stem artifacts/videos/cifar10_cosine_reverse
```

## 5. Optional Production Studio

Install the advanced dependencies:

```python
!pip install -r requirements-advanced.txt
!git clone --depth 1 https://github.com/huggingface/diffusers third_party/diffusers
```

Validate a LoRA command before starting a paid GPU run:

```python
!python3 advanced/train_lora.py data/lora/my_concept \
  --instance-prompt "a photo of sks ceramic" \
  --output-dir runs/lora/ceramic \
  --dry-run
```

Remove `--dry-run` when the image folder and command look correct.

## 6. Launch the App

```python
!python3 app.py --host 0.0.0.0 --share
```

Colab's free resources are dynamic rather than guaranteed. Save checkpoints to
Google Drive if you need them to survive runtime recycling.
