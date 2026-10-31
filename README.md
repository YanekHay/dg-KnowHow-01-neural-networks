# Neural networks, demystified

Materials for the DocGaid know-how share on neural networks: a slide deck and a live-demo notebook.
No GPU needed, and no MNIST: the demos untangle two spirals and teach a small network to paint a cat photo.

## Run the notebook

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab neural_networks_demo.ipynb
```

Six packages in total: `torch`, `numpy`, `matplotlib`, `pillow`, `ipywidgets`, `jupyterlab`.

**Smaller download (CPU only):** the default `torch` wheel on Linux and Windows bundles CUDA (about 2 GB).
If you have no NVIDIA GPU, install the CPU build first, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## CPU or GPU

The notebook picks a GPU if PyTorch can see one and falls back to the CPU otherwise.
The first cell prints which one it chose.

| | image size | neurons/layer | steps | one image run |
|---|---|---|---|---|
| CPU | 96 px | 128 | 600 | about 25-40 s |
| GPU | 256 px | 256 | 1500 | about 25-35 s |

Running every cell on an 8-thread laptop CPU takes about 3 minutes of training in total.
Slower machines scale roughly linearly. If a run feels long, lower **Steps** or **Resolution** in the widget.
The `SIDE, WIDTH, STEPS, SIGMA` line in the first code cell controls the profile.

## What's here

| File | What it is |
|---|---|
| `neural_networks_demo.ipynb` | The live demo. Models and training loops are in the notebook. |
| `nn_demo_utils.py` | Widgets and plotting, kept out of the notebook so it stays readable. |
| `images/` | Input images for the demos. Drop your own in and they show up in the dropdown. |
| `figures/` | Pre-rendered results from a GPU run, as a backup if a live run fails. |
| `Neural Networks, Demystified.pdf` | The slide deck. |

## Notebook outline

1. A neuron is tiny: a weighted sum and a squiggle.
2. Why the squiggle matters: without a nonlinearity, depth collapses into one layer.
3. Demo 1: untangling spirals, with a live decision boundary.
4. Demo 2: an image is a function `f(x, y) -> (r, g, b)`.
5. Spectral bias: why plain networks stay blurry, and how Fourier features fix it.
6. Memorization vs. generalization: hide half the pixels and watch high frequencies overfit.
7. Bonus: render the trained network at 4x resolution.
