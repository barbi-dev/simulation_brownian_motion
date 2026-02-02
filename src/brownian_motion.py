#!/usr/bin/env python3
"""
Brownian Motion 2D
Author: BarbiDev
Description:
    Simulation of 2D Brownian motion using a random walk model with Cyberpunk style.

Exports:
- assets/hero.png            
- assets/preview.gif         
- assets/brownian.mp4 

Dependencies:
pip install -r requirements.txt
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    seed: int = 77
    n_steps: int = 5000         # total random-walk steps (high = dense traces)
    sigma: float = 20.5           # step scale (std)
    hero_steps: int = 5000       # how many steps drawn in hero.png
    stride: int = 14              # downsample draw for speed (draw every k points)

    # GIF (README preview)
    gif_frames: int = 180
    gif_fps: int = 9
    gif_tail: int = 30         # trailing segment length for GIF

    # MP4 (TikTok vertical)
    mp4_frames: int = 240
    mp4_fps: int = 12
    mp4_tail: int = 30
    mp4_bitrate: int = 3000      # kbps-ish; higher = sharper neons
    mp4_name: str = "brownian7.mp4"

    # Output
    out_dir: str = "assets"
    hero_name: str = "hero7.png"
    gif_name: str = "preview7.gif"

    # Look & feel (cyberpunk)
    bg: str = "#070814"          # deep near-black with slight blue
    neon_cyan: str = "#00FFFF"
    neon_magenta: str = "#FF00FF"
    neon_purple: str = "#8A2BE2"
    neon_orange: str = "#FF4500"

    # HERO figure (square, for thumbnails / repo cover)
    hero_dpi: int = 220
    hero_fig_w: float = 10.0
    hero_fig_h: float = 10.0

    # TIKTOK figure (vertical 9:16)
    tiktok_w_px: int = 1080
    tiktok_h_px: int = 1920
    tiktok_dpi: int = 120  # 1080/120=9 in, 1920/120=16 in


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def brownian_2d(n_steps: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=sigma, size=(n_steps, 2))
    pos = np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)])
    return pos  # (n_steps+1, 2)


def cyberpunk_cmap(cfg: Config) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "barbidev_cyberpunk",
        [cfg.neon_cyan, cfg.neon_purple, cfg.neon_magenta],
        N=256,
    )


def make_segments(xy: np.ndarray) -> np.ndarray:
    return np.stack([xy[:-1], xy[1:]], axis=1)


def setup_ax(bg: str, fig_w: float, fig_h: float, dpi: int):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    return fig, ax


def compute_limits(xy: np.ndarray, pad: float = 0.1):
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    dx = max(xmax - xmin, 1e-9)
    dy = max(ymax - ymin, 1e-9)
    return (xmin - pad * dx, xmax + pad * dx, ymin - pad * dy, ymax + pad * dy)


def export_hero(cfg: Config, pos: np.ndarray) -> None:
    fig, ax = setup_ax(cfg.bg, cfg.hero_fig_w, cfg.hero_fig_h, cfg.hero_dpi)
    cmap = cyberpunk_cmap(cfg)

    xy = pos[: cfg.hero_steps + 1 : cfg.stride]
    seg = make_segments(xy)
    t = np.linspace(0, 1, len(seg))

    # Glow underlay
    lc_glow = LineCollection(
        seg, array=t, cmap=cmap,
        linewidths=4.2, alpha=0.22,
        capstyle="round", joinstyle="round", zorder=1
    )
    ax.add_collection(lc_glow)

    # Main stroke
    lc_main = LineCollection(
        seg, array=t, cmap=cmap,
        linewidths=1.5, alpha=0.96,
        capstyle="round", joinstyle="round", zorder=2
    )
    ax.add_collection(lc_main)

    # Endpoint dot
    ax.scatter(
        [xy[-1, 0]], [xy[-1, 1]],
        s=28, color=cfg.neon_orange, alpha=0.9,
        zorder=3, edgecolors="none"
    )

    x0, x1, y0, y1 = compute_limits(xy, pad=0.10)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")

    out_path = os.path.join(cfg.out_dir, cfg.hero_name)
    fig.savefig(out_path, facecolor=cfg.bg, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Exported {out_path}")


def export_gif(cfg: Config, pos: np.ndarray) -> None:
    fig, ax = setup_ax(cfg.bg, cfg.hero_fig_w, cfg.hero_fig_h, cfg.hero_dpi)
    cmap = cyberpunk_cmap(cfg)

    xy_all = pos[:: cfg.stride]
    x0, x1, y0, y1 = compute_limits(xy_all, pad=0.09)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    norm = Normalize(0.0, 1.0)

    # segmento dummy para "inicializar" el mapeo de colores
    dummy_seg = make_segments(xy_all[:2])          # (1, 2, 2)
    dummy_t = np.array([0.0])
    lc_hist_glow = LineCollection(dummy_seg, array=dummy_t, cmap=cmap, norm=norm,
                              linewidths=3, alpha=0.18, capstyle="round", joinstyle="round", zorder=1)
    lc_hist_main = LineCollection(dummy_seg, array=dummy_t, cmap=cmap, norm=norm,
                              linewidths=3, alpha=0.6, capstyle="round", joinstyle="round", zorder=2)
    lc_tail_glow = LineCollection(dummy_seg, array=dummy_t, cmap=cmap, norm=norm,
                              linewidths=8, alpha=0.1, capstyle="round", joinstyle="round", zorder=3)
    lc_tail_main = LineCollection(dummy_seg, array=dummy_t, cmap=cmap, norm=norm,
                              linewidths=2, alpha=0.98, capstyle="round", joinstyle="round", zorder=4)

    ax.add_collection(lc_hist_glow)
    ax.add_collection(lc_hist_main)
    ax.add_collection(lc_tail_glow)
    ax.add_collection(lc_tail_main)
    dot_glow = ax.scatter([], [], s=250, color=cfg.neon_orange, alpha=0.3, zorder=5, edgecolors="none")
    dot_core = ax.scatter([], [], s=75,  color=cfg.neon_orange, alpha=1.00, zorder=6, edgecolors="none")
    n = len(xy_all)
    frames = np.linspace(1, n - 1, cfg.gif_frames).astype(int)

    def update(i: int):
        idx = frames[i]
        total = n - 1
        tail_len = cfg.gif_tail

        # HISTORIA completa (se queda)
        hist = xy_all[: idx + 1]

        # TAIL (últimos tail_len puntos)
        start = max(0, idx - tail_len)
        tail = xy_all[start: idx + 1]

        if len(hist) < 2:
            return lc_hist_glow, lc_hist_main, lc_tail_glow, lc_tail_main, dot_glow, dot_core

        # --- HISTORIA (gradiente global 0..progreso) ---
        seg_h = make_segments(hist)
        t_h = np.linspace(0.0, idx / total, len(seg_h))
        lc_hist_glow.set_segments(seg_h)
        lc_hist_main.set_segments(seg_h)
        lc_hist_glow.set_array(t_h)
        lc_hist_main.set_array(t_h)

        # --- TAIL (gradiente global en el tramo del tail) ---
        seg_t = make_segments(tail)
        t0 = start / total
        t1 = idx / total
        t_t = np.linspace(t0, t1, len(seg_t))
        lc_tail_glow.set_segments(seg_t)
        lc_tail_main.set_segments(seg_t)
        lc_tail_glow.set_array(t_t)
        lc_tail_main.set_array(t_t)

        # Fuerza refresh del colormap con blit
        lc_hist_glow.changed(); lc_hist_main.changed()
        lc_tail_glow.changed(); lc_tail_main.changed()

        # Partícula fluorescente
        pos_xy = tail[-1]
        dot_glow.set_offsets([pos_xy])
        dot_core.set_offsets([pos_xy])

        return lc_hist_glow, lc_hist_main, lc_tail_glow, lc_tail_main, dot_glow, dot_core


    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=int(11000 / cfg.gif_fps), blit=True)

    out_path = os.path.join(cfg.out_dir, cfg.gif_name)
    anim.save(out_path, writer=PillowWriter(fps=cfg.gif_fps), dpi=cfg.hero_dpi)
    plt.close(fig)
    print(f"[OK] Exported {out_path}")


def export_tiktok_mp4(cfg: Config, pos: np.ndarray) -> None:
    # Exact 9:16 pixel target via inches+dpi
    fig_w_in = cfg.tiktok_w_px / cfg.tiktok_dpi
    fig_h_in = cfg.tiktok_h_px / cfg.tiktok_dpi
    fig, ax = setup_ax(cfg.bg, fig_w_in, fig_h_in, cfg.tiktok_dpi)
    cmap = cyberpunk_cmap(cfg)

    xy_all = pos[:: cfg.stride]
    n = len(xy_all)

    # Stable framing, centered, with extra pad for vertical composition
    x0, x1, y0, y1 = compute_limits(xy_all, pad=0.09)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")

    # Add a subtle vignette feel by slightly tightening the axes inside the canvas
    # (keeps edges darker; looks more "neon poster")
    ax.set_position([0.05, 0.05, 0.95, 0.95])
    norm = Normalize(0.0, 1.0)

    # segmento dummy para "inicializar" el mapeo de colores
    dummy_seg = make_segments(xy_all[:2])          # (1, 2, 2)
    dummy_t = np.array([0.0])
    
    lc_hist_glow = LineCollection(dummy_seg, array=dummy_t, cmap=cmap, norm=norm,
                              linewidths=3, alpha=0.18, capstyle="round", joinstyle="round", zorder=1)
    lc_hist_main = LineCollection(dummy_seg, array=dummy_t, cmap=cmap, norm=norm,
                              linewidths=3, alpha=0.6, capstyle="round", joinstyle="round", zorder=2)
    lc_tail_glow = LineCollection(dummy_seg, array=dummy_t, cmap=cmap, norm=norm,
                              linewidths=8, alpha=0.1, capstyle="round", joinstyle="round", zorder=3)
    lc_tail_main = LineCollection(dummy_seg, array=dummy_t, cmap=cmap, norm=norm,
                              linewidths=2, alpha=0.98, capstyle="round", joinstyle="round", zorder=4)

    ax.add_collection(lc_hist_glow)
    ax.add_collection(lc_hist_main)
    ax.add_collection(lc_tail_glow)
    ax.add_collection(lc_tail_main)

    # Halo (glow difuso)
    dot_glow = ax.scatter(
        [], [],
        s=300,                      # mucho más grande
        color=cfg.neon_orange,
        alpha=0.3,                 # muy transparente
        zorder=3,
        edgecolors="none"
    )
    dot_core = ax.scatter(
        [], [],
        s=100,                       # pequeño y definido
        color=cfg.neon_orange,
        alpha=1.0,                  # totalmente brillante
        zorder=4,
        edgecolors="none"
    )


    frames = np.linspace(1, n - 1, cfg.mp4_frames).astype(int)

    def update(i: int):
        idx = frames[i]
        total = n - 1
        tail_len = cfg.mp4_tail

        hist = xy_all[: idx + 1]
        start = max(0, idx - tail_len)
        tail = xy_all[start: idx + 1]

        if len(hist) < 2:
            return lc_hist_glow, lc_hist_main, lc_tail_glow, lc_tail_main, dot_glow, dot_core

        # HISTORIA
        seg_h = make_segments(hist)
        t_h = np.linspace(0.0, idx / total, len(seg_h))
        lc_hist_glow.set_segments(seg_h)
        lc_hist_main.set_segments(seg_h)
        lc_hist_glow.set_array(t_h)
        lc_hist_main.set_array(t_h)

        # TAIL
        seg_t = make_segments(tail)
        t0 = start / total
        t1 = idx / total
        t_t = np.linspace(t0, t1, len(seg_t))
        lc_tail_glow.set_segments(seg_t)
        lc_tail_main.set_segments(seg_t)
        lc_tail_glow.set_array(t_t)
        lc_tail_main.set_array(t_t)

        lc_hist_glow.changed(); lc_hist_main.changed()
        lc_tail_glow.changed(); lc_tail_main.changed()

        pos_xy = tail[-1]
        dot_glow.set_offsets([pos_xy])
        dot_core.set_offsets([pos_xy])
        return lc_hist_glow, lc_hist_main, lc_tail_glow, lc_tail_main, dot_glow, dot_core

    
    anim = FuncAnimation(fig,
                         update, frames=len(frames),
                         interval=int(9000 / cfg.mp4_fps), blit=True)

    out_path = os.path.join(cfg.out_dir, cfg.mp4_name)

    # FFMpegWriter settings: H.264, web-compatible pixel format
    try:
        writer = FFMpegWriter(
            fps=cfg.mp4_fps,
            codec="libx264",
            bitrate=cfg.mp4_bitrate,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        anim.save(out_path, writer=writer, dpi=cfg.tiktok_dpi)
        print(f"[OK] Exported {out_path}")
    except Exception as e:
        plt.close(fig)
        print("\n[MP4 ERROR] Could not export MP4. Likely FFmpeg is missing or not on PATH.")
        print("Install FFmpeg and try again. Error details:")
        print(str(e))
        return

    plt.close(fig)


def main():
    cfg = Config()
    ensure_dir(cfg.out_dir)

    pos = brownian_2d(cfg.n_steps, cfg.sigma, cfg.seed)

    export_hero(cfg, pos)
    export_gif(cfg, pos)
    export_tiktok_mp4(cfg, pos)


if __name__ == "__main__":
    main()
