"""UI / plotting plumbing for the neural-network demo notebook.

Nothing in here is "machine learning" - the models, encodings and training
loops live in the notebook itself so they can be shown on screen. This file
only turns tensors into pictures and wires up the ipywidgets dashboards.
"""
import io
import time
from pathlib import Path

import numpy as np
import torch
import ipywidgets as W
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap

IMAGES_DIR = Path(__file__).parent / "images"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Palette shared with the slide deck
BLUE = "#2F6FDE"
ORANGE = "#E8833A"
INK = "#1D2433"
MUTED = "#6B7385"
PAPER = "#FBFAF7"

plt_style = {
    "axes.edgecolor": "#C9CDD6",
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------
def fig_to_png(fig):
    buf = io.BytesIO()
    FigureCanvasAgg(fig)
    fig.savefig(buf, format="png", dpi=fig.dpi, facecolor=fig.get_facecolor())
    return buf.getvalue()


def arr_to_png(arr):
    """float array in [0,1], HxWx3 -> PNG bytes."""
    arr = np.clip(np.asarray(arr) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def list_images():
    return sorted(p.name for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_image(name, max_side=256):
    """Load an image from images/ as a float32 HxWx3 array in [0,1]."""
    img = Image.open(IMAGES_DIR / name)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    scale = max_side / max(img.size)
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(new_size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def psnr(mse):
    return float(-10.0 * np.log10(max(float(mse), 1e-10)))


def loss_chart(hist, title="", width=6.4, height=2.4, ylabel="PSNR (dB) - higher is better"):
    """hist: dict name -> (steps, values)."""
    import matplotlib as mpl

    with mpl.rc_context(plt_style):
        fig = Figure(figsize=(width, height), dpi=100, facecolor="white")
        ax = fig.add_subplot(111)
        colors = [BLUE, ORANGE, "#5BA88F", "#9A6BD6"]
        for (name, (xs, ys)), c in zip(hist.items(), colors):
            if len(xs):
                ax.plot(xs, ys, color=c, lw=2, label=name)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, color=INK, fontsize=11, loc="left")
        if len(hist) > 1:
            ax.legend(frameon=False, loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig_to_png(fig)


# ----------------------------------------------------------------------------
# Part 1: spirals
# ----------------------------------------------------------------------------
def make_spirals(n_per_class=400, turns=1.75, noise=0.08, seed=0):
    """Two interleaved spirals. Returns X (N,2) float32 and y (N,) int64."""
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.uniform(0, 1, n_per_class)) * turns * 2 * np.pi
    r = t / (turns * 2 * np.pi)
    xs, ys = [], []
    for label, phase in ((0, 0.0), (1, np.pi)):
        x = np.stack([r * np.cos(t + phase), r * np.sin(t + phase)], 1)
        xs.append(x + rng.normal(0, noise * 0.5, x.shape))
        ys.append(np.full(n_per_class, label))
    X = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    return X, y


SPIRAL_CMAP = ListedColormap(["#C9D9F6", "#F8D9C0"])


def spiral_frame(model, X, y, device, title="", grid_n=220, lim=1.25, dpi=100):
    import matplotlib as mpl

    xs = np.linspace(-lim, lim, grid_n, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs)
    grid = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], 1)).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(grid), 1)[:, 1].reshape(grid_n, grid_n).cpu().numpy()
    with mpl.rc_context(plt_style):
        fig = Figure(figsize=(4.6, 4.6), dpi=dpi, facecolor="white")
        ax = fig.add_subplot(111)
        ax.contourf(gx, gy, prob, levels=[0, 0.5, 1], cmap=SPIRAL_CMAP)
        ax.contour(gx, gy, prob, levels=[0.5], colors=[INK], linewidths=1.2)
        ax.scatter(*X[y == 0].T, s=9, c=BLUE, edgecolors="white", linewidths=0.3)
        ax.scatter(*X[y == 1].T, s=9, c=ORANGE, edgecolors="white", linewidths=0.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if title:
            ax.set_title(title, color=INK, fontsize=11, loc="left")
        fig.tight_layout()
        return fig_to_png(fig)


def spiral_dashboard(make_mlp, train_spirals, activations, device):
    """make_mlp / train_spirals come from the notebook."""
    X, y = make_spirals()
    act = W.ToggleButtons(options=list(activations), value="ReLU", description="Activation")
    depth = W.IntSlider(value=3, min=1, max=6, description="Hidden layers")
    width = W.IntSlider(value=32, min=2, max=128, step=2, description="Neurons/layer")
    steps = W.IntSlider(value=1500, min=100, max=5000, step=100, description="Steps")
    lr = W.FloatLogSlider(value=1e-2, base=10, min=-4, max=-0.5, description="Learning rate")
    btn = W.Button(description="Train", button_style="primary", icon="play")
    status = W.HTML()
    board = W.Image(format="png", width=460)
    chart = W.Image(format="png", width=520)

    def show_untrained(*_):
        torch.manual_seed(0)
        m = make_mlp(2, 2, width.value, depth.value, activations[act.value]).to(device)
        board.value = spiral_frame(m, X, y, device, "untrained network (random weights)")
        chart.value = loss_chart({"accuracy": ([], [])}, ylabel="train accuracy (%)", width=5.2, height=2.6)

    def run(*_):
        btn.disabled = True
        torch.manual_seed(0)
        model = make_mlp(2, 2, width.value, depth.value, activations[act.value]).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        hist_x, hist_acc = [], []
        t0 = time.time()

        def on_frame(step, loss, acc):
            hist_x.append(step)
            hist_acc.append(acc * 100)
            board.value = spiral_frame(model, X, y, device, f"step {step} - accuracy {acc*100:.1f}%")
            chart.value = loss_chart({"accuracy": (hist_x, hist_acc)}, ylabel="train accuracy (%)",
                                     width=5.2, height=2.6)
            status.value = (f"<b>{act.value}</b> · {depth.value}×{width.value} · "
                            f"{n_params:,} parameters · step {step} · loss {loss:.3f} · "
                            f"{time.time()-t0:.1f}s")

        try:
            train_spirals(model, X, y, steps.value, lr.value, on_frame)
        finally:
            btn.disabled = False

    btn.on_click(run)
    for w in (act, depth, width):
        w.observe(show_untrained, names="value")
    show_untrained()
    for w in (act, depth, width, steps, lr):
        w.style = {"description_width": "110px"}
    controls = W.VBox([act, depth, width, steps, lr, btn, status])
    return W.VBox([controls, W.HBox([board, chart])])


# ----------------------------------------------------------------------------
# Part 2: an image as a function f(x, y) -> (r, g, b)
# ----------------------------------------------------------------------------
def make_coords(h, w, device):
    """All pixel centres as (h*w, 2) tensor in [0,1]^2, row-major (y, x)."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)


def masked_view(img, train_mask):
    """Show what the network is allowed to see: hidden pixels as a checkerboard."""
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    checker = np.where(((yy // 4 + xx // 4) % 2)[..., None] == 0, 0.86, 0.78) * np.ones(3)
    m = train_mask.reshape(h, w, 1)
    return np.where(m, img, checker)


class ImageFitDashboard:
    """Interactive panel around the notebook's `fit_image` training function."""

    def __init__(self, fit_image, device, default_image=None, res=256, width=256, steps=1500, sigma=10):
        self.fit_image = fit_image
        self.device = device
        imgs = list_images()
        s = {"description_width": "110px"}
        self.image = W.Dropdown(options=imgs, value=default_image if default_image in imgs else imgs[0],
                                description="Image", style=s)
        self.res = W.IntSlider(value=res, min=64, max=512, step=32, description="Resolution", style=s)
        self.enc = W.ToggleButtons(options=["raw (x, y)", "Fourier features"], value="raw (x, y)",
                                   description="Input", style=s)
        self.sigma = W.FloatLogSlider(value=sigma, base=10, min=0, max=2.5, description="Frequency σ", style=s)
        self.width = W.IntSlider(value=width, min=16, max=512, step=16, description="Neurons/layer", style=s)
        self.depth = W.IntSlider(value=4, min=1, max=8, description="Hidden layers", style=s)
        self.steps = W.IntSlider(value=steps, min=100, max=5000, step=100, description="Steps", style=s)
        self.lr = W.FloatLogSlider(value=1e-3, base=10, min=-4, max=-1.5, description="Learning rate", style=s)
        self.holdout = W.Checkbox(value=False, description="Hide 50% of pixels (test set)")
        self.btn = W.Button(description="Train", button_style="primary", icon="play")
        self.status = W.HTML()
        self.target_view = W.Image(format="png", width=360)
        self.pred_view = W.Image(format="png", width=360)
        self.chart = W.Image(format="png", width=640)
        self.strip = W.HBox()
        self.last = None

        self.btn.on_click(self.run)
        for w in (self.image, self.res, self.holdout):
            w.observe(self.refresh_target, names="value")
        self.enc.observe(lambda *_: self._sync_sigma(), names="value")
        self._sync_sigma()
        self.refresh_target()

    def _sync_sigma(self):
        self.sigma.disabled = self.enc.value != "Fourier features"

    def _load(self):
        img = load_image(self.image.value, self.res.value)
        h, w, _ = img.shape
        rng = np.random.default_rng(0)
        mask = rng.uniform(size=h * w) < 0.5 if self.holdout.value else np.ones(h * w, bool)
        return img, mask

    def refresh_target(self, *_):
        img, mask = self._load()
        self.target_view.value = arr_to_png(masked_view(img, mask))
        self.pred_view.value = arr_to_png(np.full_like(img, 0.5))
        self.chart.value = loss_chart({"train pixels": ([], [])})
        self.strip.children = ()
        h, w, _ = img.shape
        seen = int(mask.sum())
        self.status.value = (f"{w}×{h} image → <b>{h*w:,} training examples</b> "
                             f"(one per pixel){'' if seen == h*w else f', network sees only {seen:,}'}")

    @staticmethod
    def _label(text):
        return W.HTML(f"<div style='font-weight:600;color:{INK};margin:4px 0'>{text}</div>")

    def run(self, *_):
        self.btn.disabled = True
        img, mask = self._load()
        h, w, _ = img.shape
        snaps_at = {0, 25, 100, 300}
        snaps = []
        hist = {"train pixels": ([], [])}
        if self.holdout.value:
            hist["hidden pixels (test)"] = ([], [])
        t0 = time.time()
        info = {}

        def on_frame(step, pred, train_mse, test_mse, n_params):
            info["params"] = n_params
            self.pred_view.value = arr_to_png(pred)
            hist["train pixels"][0].append(step)
            hist["train pixels"][1].append(psnr(train_mse))
            if test_mse is not None:
                hist["hidden pixels (test)"][0].append(step)
                hist["hidden pixels (test)"][1].append(psnr(test_mse))
            self.chart.value = loss_chart(hist)
            if step in snaps_at:
                snaps.append((step, pred))
            test_txt = "" if test_mse is None else f" · test PSNR <b>{psnr(test_mse):.1f} dB</b>"
            self.status.value = (f"step {step}/{self.steps.value} · train PSNR <b>{psnr(train_mse):.1f} dB</b>"
                                 f"{test_txt} · {n_params:,} parameters vs {h*w*3:,} pixel values · "
                                 f"{time.time()-t0:.1f}s")

        try:
            model, encode = self.fit_image(
                img, train_mask=mask,
                encoding="fourier" if self.enc.value == "Fourier features" else "raw",
                sigma=self.sigma.value, width=self.width.value, depth=self.depth.value,
                steps=self.steps.value, lr=self.lr.value, on_frame=on_frame)
        finally:
            self.btn.disabled = False
        self.last = dict(model=model, encode=encode, img=img, mask=mask)
        snaps.append((self.steps.value, None))
        thumbs = []
        for step, pred in snaps:
            if pred is None:
                continue
            thumbs.append(W.VBox([W.Image(value=arr_to_png(pred), format="png", width=150),
                                  W.HTML(f"<div style='text-align:center;color:{MUTED}'>step {step}</div>")]))
        self.strip.children = thumbs

    def widget(self):
        controls = W.HBox([
            W.VBox([self.image, self.res, self.enc, self.sigma]),
            W.VBox([self.width, self.depth, self.steps, self.lr, self.holdout]),
        ])
        views = W.HBox([
            W.VBox([self._label("What the network is trained on"), self.target_view]),
            W.VBox([self._label("What the network has learned"), self.pred_view]),
        ])
        return W.VBox([controls, W.HBox([self.btn]), self.status, views, self.chart,
                       self._label("Learning progress"), self.strip])


def compare_grid(fit_image, img, configs, device, mask=None, steps=1500, thumb=300):
    """Train several configurations one after another and show them side by side.

    configs: list of (label, kwargs-for-fit_image)
    """
    h, w, _ = img.shape
    if mask is None:
        mask = np.ones(h * w, bool)
    tiles, captions = [], []
    target_tile = W.VBox([W.Image(value=arr_to_png(masked_view(img, mask)), format="png", width=thumb),
                          W.HTML(f"<div style='text-align:center;color:{INK};font-weight:600'>target</div>")])
    for label, _ in configs:
        tiles.append(W.Image(value=arr_to_png(np.full_like(img, 0.5)), format="png", width=thumb))
        captions.append(W.HTML(f"<div style='text-align:center;color:{INK};font-weight:600'>{label}</div>"))
    box = W.HBox([target_tile] + [W.VBox([t, c]) for t, c in zip(tiles, captions)])
    from IPython.display import display
    display(box)
    results = []
    for (label, kw), tile, cap in zip(configs, tiles, captions):
        def on_frame(step, pred, train_mse, test_mse, n_params, tile=tile, cap=cap, label=label):
            tile.value = arr_to_png(pred)
            txt = f"train {psnr(train_mse):.1f} dB"
            if test_mse is not None:
                txt += f" · test {psnr(test_mse):.1f} dB"
            cap.value = (f"<div style='text-align:center;color:{INK};font-weight:600'>{label}</div>"
                         f"<div style='text-align:center;color:{MUTED}'>{txt}</div>")
        model, encode = fit_image(img, train_mask=mask, steps=steps, on_frame=on_frame, **kw)
        results.append((label, model, encode))
    return results
