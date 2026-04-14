# DROID-W Installation Notes

This document records the installation process that was actually used to run DROID-W successfully on this machine.

Environment used:

- GPU: NVIDIA GeForce RTX 5090
- CUDA toolkit on system: 12.8
- Python: 3.10 via `uv`
- PyTorch: `2.7.0+cu128`
- xformers: `0.0.33+aa7bc366.d20260414`
- mmcv: `1.7.2`

## Why this differs from `README.md`

The upstream `README.md` assumes:

- `conda`
- `torch==2.1.0`
- `cu118`

That stack is not suitable for this machine. Since this server uses RTX 5090 (`sm_120`) and CUDA 12.8, the working setup uses:

- `uv` instead of `conda`
- `torch==2.7.0+cu128`
- source-built `xformers`
- several compatibility patches in this repo for PyTorch 2.7 and `sm_120`

## 1. Create the environment

```bash
cd ~/DROID-W
uv venv --python 3.10 .venv
source .venv/bin/activate
```

## 2. Install PyTorch for CUDA 12.8

```bash
uv pip install --python .venv/bin/python \
  numpy==1.26.3 \
  torch==2.7.0 \
  torchvision \
  torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

## 3. Install build helpers

```bash
uv pip install --python .venv/bin/python ninja setuptools wheel packaging
```

## 4. Install `torch-scatter`

```bash
uv pip install --python .venv/bin/python \
  torch-scatter \
  -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

## 5. Install `mmcv`

`Metric3D` imports `mmcv.utils.*`, so `mmcv` is required.

```bash
uv pip install --python .venv/bin/python --no-build-isolation mmcv==1.7.2
```

## 6. Build and install `xformers` from source

The prebuilt wheel we tried did not include `sm_120`, so it failed on RTX 5090 with:

`no kernel image is available for execution on the device`

We solved this by building `xformers` from source from a local clone at tag `v0.0.33`.

Important: before building from source, run:

```bash
git submodule update --init --recursive
```

Working commands:

```bash
cd ~/xformers
git checkout v0.0.33
git submodule update --init --recursive

uv pip uninstall --python ~/DROID-W/.venv/bin/python xformers

TORCH_CUDA_ARCH_LIST="12.0" \
MAX_JOBS=4 \
XFORMERS_BUILD_TYPE=Release \
uv pip install \
  --python ~/DROID-W/.venv/bin/python \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  -v \
  ~/xformers
```

If your compiler toolchain complains, retry with:

```bash
TORCH_CUDA_ARCH_LIST="12.0" \
MAX_JOBS=4 \
XFORMERS_BUILD_TYPE=Release \
NVCC_FLAGS="-allow-unsupported-compiler" \
uv pip install \
  --python ~/DROID-W/.venv/bin/python \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  -v \
  ~/xformers
```

Verify:

```bash
python -m xformers.info
```

The important line is:

- `gpu.compute_capability: 12.0`

Also check that the build arch list includes `12.0`.

## 7. Install the vendored CUDA extensions

Install in this order:

```bash
cd ~/DROID-W

uv pip install --python .venv/bin/python -e thirdparty/lietorch --no-build-isolation
uv pip install --python .venv/bin/python -e thirdparty/diff-gaussian-rasterization-w-pose --no-build-isolation
uv pip install --python .venv/bin/python -e thirdparty/simple-knn --no-build-isolation
uv pip install --python .venv/bin/python -e . --no-build-isolation
```

## 8. Install the remaining Python dependencies

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

## 9. Compatibility changes applied in this repo

This repo needed several source-level fixes to build and run correctly with:

- `torch 2.7`
- `cuda 12.8`
- `RTX 5090 / sm_120`

Main changes:

- `setup.py`
  - added `sm_120`/`compute_120`
  - added `-DEIGEN_NO_CUDA`
- `thirdparty/lietorch/setup.py`
  - added `sm_120`/`compute_120`
- `src/lib/altcorr_kernel.cu`
  - replaced deprecated `tensor.type()` usage
- `src/lib/correlation_kernels.cu`
  - replaced deprecated `tensor.type()` usage
- `thirdparty/lietorch/lietorch/include/dispatch.h`
  - updated dispatch handling for current PyTorch
- `thirdparty/lietorch/lietorch/src/lietorch_gpu.cu`
  - replaced deprecated `tensor.type()` usage
- `thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp`
  - replaced deprecated `tensor.type()` usage
- `thirdparty/lietorch/lietorch/extras/corr_index_kernel.cu`
  - replaced deprecated `tensor.type()` usage
- `thirdparty/lietorch/lietorch/extras/extras.cpp`
  - replaced deprecated CUDA tensor check
- `thirdparty/diff-gaussian-rasterization-w-pose/cuda_rasterizer/rasterizer_impl.h`
  - added missing standard headers for integer types
- `thirdparty/simple-knn/simple_knn.cu`
  - added missing `<cfloat>`
- `thirdparty/simple-knn/simple_knn/__init__.py`
  - added package init so `import simple_knn` works

## 10. Verify the core imports

```bash
python -c "import torch, lietorch, simple_knn, diff_gaussian_rasterization, droid_backends, torch_scatter, xformers; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected result should include:

- `2.7.0+cu128`
- `12.8`
- `True`

## 11. Pretrained checkpoint

Place the checkpoint here:

```text
pretrained/droid.pth
```

## 12. Example validation run

For the Bonn balloon sequence:

```bash
python scripts/validate_bonn_balloon.py --device cuda:0
```

This writes results under:

```text
Outputs/Bonn/bonn_balloon
```

Useful output files:

- `Outputs/Bonn/bonn_balloon/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_balloon/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_balloon/traj/est_poses_full.txt`
- `Outputs/Bonn/bonn_balloon/video.npz`
