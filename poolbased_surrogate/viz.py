"""Figure helpers for logging: colormaps, heatmaps, space-time grids, rollout plots.

matplotlib/PIL are imported lazily inside the one function that needs them so the
package stays importable without them.
"""
from __future__ import annotations

import io

import numpy as np


def colormap_rgb(values: np.ndarray, cmap: str) -> np.ndarray:
    anchors = {
        "magma": np.asarray(
            [
                [0.001, 0.000, 0.014],
                [0.232, 0.060, 0.438],
                [0.550, 0.161, 0.506],
                [0.868, 0.287, 0.409],
                [0.995, 0.746, 0.518],
                [0.988, 0.998, 0.645],
            ],
            dtype=np.float32,
        ),
        "viridis": np.asarray(
            [
                [0.267, 0.005, 0.329],
                [0.283, 0.141, 0.458],
                [0.254, 0.265, 0.530],
                [0.207, 0.372, 0.553],
                [0.164, 0.471, 0.558],
                [0.128, 0.567, 0.551],
                [0.135, 0.659, 0.518],
                [0.267, 0.749, 0.441],
                [0.478, 0.821, 0.318],
                [0.741, 0.873, 0.150],
                [0.993, 0.906, 0.144],
            ],
            dtype=np.float32,
        ),
        "coolwarm": np.asarray(
            [
                [0.230, 0.299, 0.754],
                [0.554, 0.690, 0.996],
                [0.867, 0.864, 0.863],
                [0.957, 0.598, 0.477],
                [0.706, 0.016, 0.150],
            ],
            dtype=np.float32,
        ),
        "icefire": np.asarray(
            [
                [0.094, 0.110, 0.262],
                [0.050, 0.379, 0.596],
                [0.735, 0.925, 0.929],
                [0.973, 0.730, 0.489],
                [0.690, 0.083, 0.165],
                [0.122, 0.043, 0.084],
            ],
            dtype=np.float32,
        ),
    }[cmap]
    scaled = np.asarray(values, dtype=np.float32).clip(0.0, 1.0)
    pos = scaled * (len(anchors) - 1)
    left = np.floor(pos).astype(np.int64)
    right = np.clip(left + 1, 0, len(anchors) - 1)
    weight = (pos - left)[..., None]
    rgb = anchors[left] * (1.0 - weight) + anchors[right] * weight
    return (255.0 * rgb).clip(0, 255).astype(np.uint8)


def heatmap_image(
    values: np.ndarray,
    cmap: str = "coolwarm",
    center_zero: bool = True,
    scale: float | None = None,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if center_zero:
        signed_scale = float(scale) if scale is not None else float(np.quantile(np.abs(values), 0.98))
        signed_scale = max(signed_scale, 1e-8)
        scaled = 0.5 + 0.5 * (values / signed_scale)
    else:
        if value_range is None:
            lo, hi = np.quantile(values, [0.02, 0.98])
        else:
            lo, hi = value_range
        scaled = (values - lo) / (hi - lo + 1e-8)
    return colormap_rgb(scaled, cmap)


def stacked_example_image(examples: dict[str, np.ndarray]) -> np.ndarray:
    rows = []
    for i in range(examples["input"].shape[0]):
        rows.extend(
            [
                examples["input"][i],
                examples["target"][i],
                examples["prediction"][i],
                examples["error"][i],
            ]
        )
    return heatmap_image(np.stack(rows, axis=0), cmap="coolwarm", center_zero=True)


def rollout_error_pyplot_image(series: dict[str, np.ndarray], metric: str) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    step = series["step"]
    for key, linestyle in (("mean", "-"), ("p95", "--"), ("p99", ":")):
        values = series[f"{metric}_{key}"]
        ax.plot(step, values, linestyle=linestyle, linewidth=2.0, label=key)
    ax.set_title(f"Rollout {metric.upper()} per step")
    ax.set_xlabel("rollout step")
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


def spacetime_grid_image(spacetime: dict[str, np.ndarray], n_rows: int = 3) -> np.ndarray:
    """Build a single (n_rows x 3) space-time grid image: each row is one example,
    columns are [ground truth | prediction | absolute error]. Per-row color scale is
    anchored to that example's ground-truth q98 amplitude. Rows are separated by borders.
    """
    n_rows = int(min(n_rows, spacetime["truth"].shape[0]))
    rows = []
    row_w = None
    for i in range(n_rows):
        gt_scale = max(float(np.quantile(np.abs(spacetime["truth"][i]), 0.98)), 1e-8)
        gt_rgb = heatmap_image(spacetime["truth"][i], cmap="coolwarm", center_zero=True, scale=gt_scale)
        H, W, _ = gt_rgb.shape
        indices = np.round(np.linspace(0, W - 1, H)).astype(np.int64)
        gt_sq = gt_rgb[:, indices, :]
        pred_sq = heatmap_image(
            spacetime["prediction"][i], cmap="coolwarm", center_zero=True, scale=gt_scale
        )[:, indices, :]
        abs_err_sq = heatmap_image(
            np.abs(spacetime["error"][i]), cmap="magma", center_zero=False, value_range=(0.0, gt_scale)
        )[:, indices, :]
        bw = max(2, H // 100)
        vborder = np.zeros((H, bw, 3), dtype=np.uint8)
        row = np.concatenate([gt_sq, vborder, pred_sq, vborder, abs_err_sq], axis=1)
        rows.append(row)
        row_w = row.shape[1]
    hborder = np.zeros((max(2, rows[0].shape[0] // 100), row_w, 3), dtype=np.uint8)
    stacked = []
    for i, r in enumerate(rows):
        if i > 0:
            stacked.append(hborder)
        stacked.append(r)
    return np.concatenate(stacked, axis=0)


def state_profile_grid_image(
    states: np.ndarray,
    max_examples: int = 8,
    width: int = 900,
    row_height: int = 112,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 3:
        states = states[:, 0]
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    n = min(max_examples, len(states))
    if n == 0:
        return np.full((row_height, width, 3), 255, dtype=np.uint8)

    idx = np.linspace(0, len(states) - 1, n, dtype=np.int64)
    profiles = states[idx]
    scale = float(np.quantile(np.abs(profiles), 0.98))
    scale = max(scale, 1e-8)

    pad_x = 18
    pad_y = 12
    height = n * row_height
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x = np.linspace(pad_x, width - pad_x - 1, profiles.shape[1])

    def draw_line(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int]):
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, steps).round().astype(np.int64)
        ys = np.linspace(y0, y1, steps).round().astype(np.int64)
        ok = (xs >= 0) & (xs < img.shape[1]) & (ys >= 0) & (ys < img.shape[0])
        img[ys[ok], xs[ok]] = color

    line_colors = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
    ]
    for row, profile in enumerate(profiles):
        top = row * row_height
        bottom = top + row_height - 1
        center = top + row_height // 2
        usable = row_height - 2 * pad_y
        y = center - np.clip(profile / scale, -1.0, 1.0) * (usable / 2.0)

        image[top:bottom, pad_x] = (210, 210, 210)
        image[top:bottom, width - pad_x - 1] = (210, 210, 210)
        image[center, pad_x : width - pad_x] = (220, 220, 220)
        image[top + pad_y, pad_x : width - pad_x] = (235, 235, 235)
        image[bottom - pad_y, pad_x : width - pad_x] = (235, 235, 235)

        color = line_colors[row % len(line_colors)]
        for k in range(len(x) - 1):
            draw_line(image, x[k], y[k], x[k + 1], y[k + 1], color)
    return image


def state_profile_line_series(states: np.ndarray, max_examples: int = 8, max_points: int = 400):
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 3:
        states = states[:, 0]
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    n = min(max_examples, len(states))
    if n == 0:
        return [0], [[0.0]], ["empty"]
    sample_idx = np.linspace(0, len(states) - 1, n, dtype=np.int64)
    grid_idx = np.linspace(0, states.shape[1] - 1, min(max_points, states.shape[1]), dtype=np.int64)
    xs = grid_idx.tolist()
    ys = [states[i, grid_idx].astype(float).tolist() for i in sample_idx]
    keys = [f"sample_{j}" for j in range(n)]
    return xs, ys, keys
