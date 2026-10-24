# Train MNIST inside D3OS

Use this procedure to train a convolutional classifier with Burn on the native NVIDIA backend.
The host stages the dataset and precompiled kernels. D3OS performs training, evaluation, checkpoint creation, and inference export.

The model uses two convolutions, max pooling, and two dense layers with ten output logits.
Its eight parameter tensors use FP32 storage. Matrix products use compensated TF32 Tensor Core operations.
Input normalization is `(pixel / 255 - 0.1307) / 0.3081` for 28-by-28 grayscale images.

## Prepare the host

Complete the [VFIO setup](nvidia-vfio.howto.md) and native kernel preparation first.
Keep the GPU bound to `vfio-pci`. Run one GPU VM at a time.
The commands below use a 4 GiB guest and the firmware and kernel bundles under `target/nvidia/`.
Build the matching production image:

```bash
cargo make --no-workspace --profile production image
```

The current SDK uses compute protocol version 18. Rebuild applications and the kernel together.
This change adds memory reporting and does not change the device kernel binaries.

## Train and save

From the repository root, run five initial epochs, followed by two epochs at a smaller learning rate:

```bash
python3 tools/nvidia/run-mnist.py \
  --epochs 7 --max-steps 4690 \
  --output target/nvidia/burn/mnist-initial

python3 tools/nvidia/run-mnist.py \
  --resume target/nvidia/burn/mnist-initial/mnist.checkpoint \
  --epochs 7 --learning-rate 0.0001 --override-learning-rate
```

At batch size 64, five complete epochs require 4,690 steps.
The first command reports `complete=false` because it stops before the total epoch limit.
The second command finishes training and applies the final accuracy requirement.

The script downloads the original MNIST IDX archives from the mirror used by [torchvision](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py).
It verifies the published archive checksums and records SHA-256 values for the extracted data.
The dataset manifest is `target/mnist/manifest.json`. The host does not train model weights.

Defaults are seed 42, batch size 64, Adam, and learning rate 0.001.
Artifact version 1 fixes Adam beta values at 0.9 and 0.999, with epsilon `1e-5` and AMSGrad disabled.
Training uses all 60,000 images each epoch, including the final partial batch.
Evaluation uses the separate 10,000-image test split.
A complete run requires at least 99 percent test accuracy. A lower score preserves artifacts and reports failure.

The output directory defaults to `target/nvidia/burn/mnist/`:

| File | Contents |
| --- | --- |
| `mnist.checkpoint` | Named weights, Adam state, progress, shuffle state, and native GPU random state |
| `mnist.model` | Named weights, architecture, and preprocessing for inference |
| `serial.log` | Guest training, evaluation, and export records |
| `validation.json` | D3OS results and artifact transfer checks |
| `vm.log` | Host VM launcher output |

The runner exports checkpoints every 256 steps, after each epoch, and at completion.
The host verifies ordered chunks, byte counts, and CRC32 checksums before replacing an artifact.
An interrupted transfer preserves the last complete checkpoint.
These exported files survive VM shutdown. Files staged under `/fwcfg/` are read-only inputs.

To check setup without completing training, stop at a global step:

```bash
python3 tools/nvidia/run-mnist.py --max-steps 4 --output target/nvidia/burn/mnist-smoke
```

This run reports `complete=false`. It validates input, training, checkpoint export, and reloaded inference.
It does not establish the accuracy target.

## Resume in a new VM

Use the same seed, batch size, and learning rate as the saved checkpoint:

```bash
python3 tools/nvidia/run-mnist.py \
  --resume target/nvidia/burn/mnist/mnist.checkpoint \
  --epochs 10 --learning-rate 0.0001 \
  --output target/nvidia/burn/mnist-resumed
```

The epoch limit is a total, not an additional epoch count.
The checkpoint validates the dataset fingerprint, configuration, sample offset, and step count before loading.
For the two-stage recipe above, include `--learning-rate 0.0001` when resuming its final checkpoint.
The current epoch's order is reconstructed from the seed and epoch, then checked against its saved shuffle state.
Adam state loads directly onto the selected GPU. Native random state restores after model and optimizer loading.

To continue at a smaller learning rate, request that change explicitly:

```bash
python3 tools/nvidia/run-mnist.py \
  --resume target/nvidia/burn/mnist/mnist.checkpoint \
  --epochs 9 --learning-rate 0.00005 --override-learning-rate \
  --output target/nvidia/burn/mnist-refined
```

This preserves Adam moments, step counts, sample order, and random state.
The next checkpoint records the new rate. Other configuration differences remain errors.
Subsequent resumes must specify the saved rate.


## Reload the inference model

Start a fresh VM with the exported artifact:

```bash
python3 tools/nvidia/run-mnist.py \
  --model target/nvidia/burn/mnist/mnist.model \
  --output target/nvidia/burn/mnist-inference
```

Compare `test_correct` and `predictions_crc32` with the final training result.
The training VM also checks exported logits within `2e-6` and predictions across the complete test split.
The public loading API is `d3os_ai::mnist::artifact::load(bytes, device)`.
It returns the same Burn `Mnist` model type used for training.

## Read the results

Every guest record identifies D3OS and the NVIDIA backend.
`segment_loss` averages samples processed since the last epoch report or the current VM's resume boundary.
`samples_per_second` includes time since the training loop started, including evaluation and serial export delays.
It is not an isolated GPU kernel measurement or a Linux comparison.

Memory records sample live process compute buffers and the device allocation cache after a training step.
They exclude fixed firmware and channel memory and do not measure temporary peaks between samples.
Prediction vectors are explicit inference outputs. Accuracy counting also runs on the GPU and is checked against the labels.

The host tests validate IDX boundaries, sample order, epoch rollover, Adam resume, and artifact corruption.
The physical GPU self-test repeats convolutional checkpoint resume across a partial batch and epoch boundary.
It also verifies native random-state restoration, inference export, and memory-query pointer guards.

## Retained D3OS validation

The physical RTX 4090 Laptop GPU run completed seven epochs in FP32 with batch size 64 and seed 42.
The first five epochs used learning rate 0.001. Two additional epochs used 0.0001 without resetting Adam state.
The five-epoch model scored 98.93 percent. The final model scored **99.26 percent**, with 9,926 correct predictions out of 10,000.
The final epoch's mean training loss was 0.006849.

The training VM restored an exported checkpoint from an earlier VM.
Its final model passed complete test-set prediction checks and a `2e-6` logit check after inference export.
A fresh VM loads only the exported inference model and reproduces all 10,000 predictions.
Both VMs report prediction CRC32 `ce40d6c3` and 9,926 correct classifications.
Live compute buffers remained at 626,688 bytes at the reported training boundaries; the device cache reached 35,594,240 bytes.
These are boundary samples, not total board memory or peak activation memory.

Logs and artifacts are under `target/nvidia/burn/training/`.
The `smoke`, `full`, and `continued` directories preserve the run sequence, including the initial accuracy failure.
This is a D3OS correctness result. It includes no Linux training measurement or performance-parity claim.
