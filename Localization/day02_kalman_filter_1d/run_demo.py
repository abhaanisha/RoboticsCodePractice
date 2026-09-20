"""
Day 2 demo - a vehicle on a straight road with a noisy satellite receiver.

Simulates a vehicle driving along a line. It is commanded to move a fixed
distance each second, its wheels do not obey exactly, and a receiver reports a
noisy absolute position. A one dimensional Kalman filter fuses the two.

Outputs, into ./media
    kalman_filter_1d.gif          the main animation
    belief_ridge_3d.gif           the belief drawn as a 3D ridge over time
    kalman_gain_surface.png       the gain as a surface over the two noises
    variance_convergence_3d.png   convergence from every starting guess
    tracking_summary.png          the whole run as one still image

Run   python run_demo.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from PIL import Image

from kalman_filter_1d import (Gaussian, predict, update, kalman_gain,
                              steady_state_variance, steady_state_gain)

# --------------------------------------------------------------------------
# The scenario
# --------------------------------------------------------------------------
N_STEPS = 18
COMMAND = 2.0          # metres per step the vehicle is told to travel
SIGMA_PROCESS = 0.8    # metres, how badly the wheels obey the command
SIGMA_SENSOR = 3.0     # metres, the accuracy of the satellite receiver
SIGMA_INITIAL = 10.0   # metres, how vaguely the start position is known
START = 0.0
SEED = 11

Q = SIGMA_PROCESS ** 2
R = SIGMA_SENSOR ** 2

# --------------------------------------------------------------------------
# Palette, carried over from day 1 so the two chapters read as one set
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e1"
BELIEF = "#2a78d6"     # the estimate
BELIEF_LT = "#cde2fb"
TRUTH = "#eb6834"      # ground truth, known to us and not to the filter
SENSOR = "#1baf7a"     # the satellite receiver
SENSOR_DK = "#12805a"
PREDICT_BG = "#fdeee7"
UPDATE_BG = "#e8f1fd"

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
@dataclass
class Phase:
    kind: str              # "start" | "predict" | "update"
    step: int
    before: Gaussian
    after: Gaussian
    truth: float
    measurement: float | None = None
    gain: float | None = None


def simulate() -> list[Phase]:
    rng = np.random.default_rng(SEED)
    motion = Gaussian(COMMAND, Q)

    x = START
    belief = Gaussian(START, SIGMA_INITIAL ** 2)
    phases = [Phase("start", 0, belief, belief, x)]

    for k in range(1, N_STEPS + 1):
        x += COMMAND + rng.normal(0.0, SIGMA_PROCESS)
        prior = predict(belief, motion)
        phases.append(Phase("predict", k, belief, prior, x))

        z = x + rng.normal(0.0, SIGMA_SENSOR)
        posterior, K = update(prior, Gaussian(z, R))
        phases.append(Phase("update", k, prior, posterior, x, measurement=z, gain=K))
        belief = posterior

    return phases


def validate_against_filterpy(phases: list[Phase]) -> float:
    """Re-run the identical data through filterpy and compare, element by
    element. If the two disagree, the implementation here is wrong."""
    from filterpy.kalman import KalmanFilter

    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[START]])
    kf.P = np.array([[SIGMA_INITIAL ** 2]])
    kf.F = np.array([[1.0]])
    kf.H = np.array([[1.0]])
    kf.B = np.array([[1.0]])
    kf.Q = np.array([[Q]])
    kf.R = np.array([[R]])

    worst = 0.0
    for p in phases:
        if p.kind == "predict":
            kf.predict(u=np.array([[COMMAND]]))
        elif p.kind == "update":
            kf.update(np.array([[p.measurement]]))
        else:
            continue
        worst = max(worst,
                    abs(kf.x[0, 0] - p.after.mean),
                    abs(kf.P[0, 0] - p.after.var))
    return worst


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def blend(a: Gaussian, b: Gaussian, t: float) -> Gaussian:
    return Gaussian(a.mean + t * (b.mean - a.mean), a.var + t * (b.var - a.var))


def save_gif(frames, out_path: Path, fps: int, colors: int = 96):
    """Quantise every frame against one shared palette, then write the GIF."""
    sample = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    for k, f in enumerate((frames[0], frames[len(frames) // 2], frames[-1])):
        sample.paste(f, (0, k * frames[0].height))
    palette = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    quant = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quant[0].save(out_path, save_all=True, append_images=quant[1:],
                  duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    return out_path


def grab(fig) -> Image.Image:
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])


def style_axes(ax, grid_axis="both"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)


# --------------------------------------------------------------------------
# Figure 1, the main animation
# --------------------------------------------------------------------------
def make_main_gif(phases: list[Phase], out_path: Path, fps=14, ease=4, hold=2,
                  max_step=12):
    # The filter has fully settled by step 8, so animating all 18 steps only
    # makes a longer file saying the same thing.
    phases = [p for p in phases if p.step <= max_step]
    truths = [p.truth for p in phases]
    xs = np.linspace(min(truths) - 3.2 * SIGMA_INITIAL,
                     max(truths) + 3.2 * SIGMA_SENSOR, 800)
    peak = Gaussian(0.0, steady_state_variance(Q, R)).pdf(0.0) * 1.15

    fig = plt.figure(figsize=(11.0, 6.9), dpi=78, facecolor=SURFACE)
    ax_d = fig.add_axes([0.075, 0.495, 0.885, 0.325])
    ax_t = fig.add_axes([0.075, 0.145, 0.885, 0.265])
    style_axes(ax_d, "y")
    style_axes(ax_t)

    fig.text(0.075, 0.955, "Kalman Filter in One Dimension", fontsize=15.5,
             fontweight="bold", color=INK, va="center")
    fig.text(0.962, 0.955, "Day 2 / Localization", fontsize=9.5, color=INK_3,
             va="center", ha="right")
    badge = fig.text(0.075, 0.885, " PREDICT ", fontsize=9.5, fontweight="bold",
                     color=TRUTH, va="center", family="monospace",
                     bbox=dict(boxstyle="round,pad=0.42", fc=PREDICT_BG, ec="none"))
    caption = fig.text(0.183, 0.885, "", fontsize=11, color=INK_2, va="center")

    # distribution panel
    prior_line, = ax_d.plot([], [], color=BELIEF, lw=1.6, ls=(0, (4, 2.5)),
                            zorder=4, label="belief before this step")
    meas_fill = ax_d.fill_between(xs, 0, 0, color=SENSOR, alpha=0.20, zorder=2)
    meas_line, = ax_d.plot([], [], color=SENSOR_DK, lw=2.0, zorder=5,
                           label="what the receiver reported")
    post_fill = ax_d.fill_between(xs, 0, 0, color=BELIEF, alpha=0.28, zorder=3)
    post_line, = ax_d.plot([], [], color=BELIEF, lw=2.6, zorder=6,
                           label="belief after this step")
    truth_line = ax_d.axvline(0.0, color=TRUTH, lw=2.0, zorder=7)
    ax_d.set_xlim(xs[0], xs[-1])
    ax_d.set_ylim(0, peak)
    ax_d.set_yticks([])
    ax_d.set_ylabel("how likely", fontsize=9.5, color=INK_2, labelpad=8)
    ax_d.set_xlabel("position along the road in metres", fontsize=9.5,
                    color=INK_2, labelpad=2)

    # tracking panel
    ax_t.set_xlim(-0.5, max_step + 0.5)
    ax_t.set_ylim(min(truths) - 9, max(truths) + 11)
    ax_t.set_xticks(range(0, max_step + 1, 2))
    ax_t.set_xlabel("step", fontsize=9.5, color=INK_2, labelpad=2)
    ax_t.set_ylabel("position in metres", fontsize=9.5, color=INK_2)
    band = ax_t.fill_between([], [], [], color=BELIEF, alpha=0.18, zorder=2)
    true_l, = ax_t.plot([], [], color=TRUTH, lw=2.2, zorder=5)
    est_l, = ax_t.plot([], [], color=BELIEF, lw=2.0, zorder=4)
    meas_pts, = ax_t.plot([], [], ls="none", marker="o", ms=7.0, color=SENSOR,
                          markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=6)

    fig.legend(handles=[
        Line2D([], [], color=TRUTH, lw=2.2, label="where the vehicle really is"),
        Line2D([], [], ls="none", marker="o", ms=7.5, color=SENSOR,
               markeredgecolor=SURFACE, label="satellite receiver reading"),
        Line2D([], [], color=BELIEF, lw=2.2, label="the filter estimate"),
        Line2D([], [], color=BELIEF, lw=7, alpha=0.18,
               label="the range the filter considers likely"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.093), frameon=False, ncol=4,
        fontsize=8.5, labelcolor=INK_2, handletextpad=0.5, columnspacing=1.9)

    footer = fig.text(0.5, 0.024, "", fontsize=9.5, color=INK_2, va="center",
                      ha="center", family="monospace")

    frames: list[Image.Image] = []
    hist_k, hist_t, hist_mu, hist_lo, hist_hi = [], [], [], [], []
    meas_k, meas_z = [], []

    def draw(phase, belief, show_meas):
        nonlocal meas_fill, post_fill, band
        post_line.set_data(xs, belief.pdf(xs))
        post_fill.remove()
        post_fill = ax_d.fill_between(xs, 0, belief.pdf(xs), color=BELIEF,
                                      alpha=0.28, zorder=3)
        if phase.kind == "update":
            prior_line.set_data(xs, phase.before.pdf(xs))
        else:
            prior_line.set_data([], [])

        meas_fill.remove()
        if show_meas and phase.measurement is not None:
            mz = Gaussian(phase.measurement, R)
            meas_line.set_data(xs, mz.pdf(xs))
            meas_fill = ax_d.fill_between(xs, 0, mz.pdf(xs), color=SENSOR,
                                          alpha=0.20, zorder=2)
        else:
            meas_line.set_data([], [])
            meas_fill = ax_d.fill_between(xs, 0, 0, color=SENSOR, alpha=0.20, zorder=2)
        truth_line.set_xdata([phase.truth, phase.truth])

        true_l.set_data(hist_k, hist_t)
        est_l.set_data(hist_k, hist_mu)
        meas_pts.set_data(meas_k, meas_z)
        band.remove()
        band = ax_t.fill_between(hist_k, hist_lo, hist_hi, color=BELIEF,
                                 alpha=0.18, zorder=2)

        g = f"{phase.gain:.3f}" if phase.gain is not None else "  -  "
        footer.set_text(
            f"estimate {belief.mean:7.2f} m    uncertainty (1 sigma) {belief.std:5.2f} m"
            f"    truth {phase.truth:7.2f} m    error {belief.mean - phase.truth:+6.2f} m"
            f"    gain K {g}")

    for idx, phase in enumerate(phases):
        if phase.kind == "start":
            btxt, bcol, bbg = " START   ", INK_2, "#efeeea"
            cap = ("The vehicle is somewhere near the origin. A wide bell curve "
                   "says so, and says how vaguely.")
        elif phase.kind == "predict":
            btxt, bcol, bbg = " PREDICT ", TRUTH, PREDICT_BG
            cap = (f"step {phase.step}. Drive {COMMAND:.0f} m forward. Slide the curve "
                   f"along and widen it, because the wheels are not exact.")
        else:
            btxt, bcol, bbg = " UPDATE  ", BELIEF, UPDATE_BG
            cap = (f"step {phase.step}. The receiver reports a position. Multiply the two "
                   f"curves. The result is narrower than either one.")
        badge.set_text(btxt)
        badge.set_color(bcol)
        badge.get_bbox_patch().set_facecolor(bbg)
        caption.set_text(cap)

        if phase.kind == "update":
            hist_k.append(phase.step)
            hist_t.append(phase.truth)
            hist_mu.append(phase.after.mean)
            lo, hi = phase.after.interval(2.0)
            hist_lo.append(lo)
            hist_hi.append(hi)
            meas_k.append(phase.step)
            meas_z.append(phase.measurement)
        n_ease = 1 if phase.kind == "start" else ease
        n_hold = hold + (8 if phase.kind == "start" else 0)
        if idx == len(phases) - 1:
            n_hold += 20

        for f in range(n_ease):
            t = smoothstep((f + 1) / n_ease)
            draw(phase, blend(phase.before, phase.after, t), phase.kind == "update")
            frames.append(grab(fig))
        frames.extend([frames[-1]] * n_hold)

    plt.close(fig)
    # 192 colours, not the default 96. Small green marks were being quantised
    # into a blue-green blend against the blue band underneath them.
    return save_gif(frames, out_path, fps, colors=192), len(frames)


# --------------------------------------------------------------------------
# Figure 2, the belief drawn as a ridge in three dimensions
# --------------------------------------------------------------------------
def make_ridge_gif(phases: list[Phase], out_path: Path, fps=11):
    # Row 0 is the belief at switch-on, so the collapse from vague to sharp is
    # part of the picture rather than something that happened before it starts.
    rows = [p for p in phases if p.kind in ("start", "update")]
    truths = [p.truth for p in rows]
    steps = [p.step for p in rows]
    xs = np.linspace(-3.1 * SIGMA_INITIAL, max(truths) + 11, 420)
    zmax = max(p.after.pdf(p.after.mean) for p in rows) * 1.05

    n = len(rows)
    verts, colors = [], []
    for i, p in enumerate(rows):
        y = p.after.pdf(xs)
        verts.append([(xs[0], 0.0), *zip(xs, y), (xs[-1], 0.0)])
        colors.append(BLUE_RAMP(0.10 + 0.90 * i / max(n - 1, 1)))

    frames = []
    for f in range(n + 14):
        k = min(f + 1, n)
        fig = plt.figure(figsize=(9.6, 6.2), dpi=82, facecolor=SURFACE)
        ax = fig.add_axes([0.02, 0.02, 0.90, 0.80], projection="3d",
                          facecolor=SURFACE)

        poly = PolyCollection(verts[:k], facecolors=colors[:k],
                              edgecolors=[(1, 1, 1, 0.85)] * k, linewidths=0.9)
        poly.set_alpha(0.86)
        ax.add_collection3d(poly, zs=steps[:k], zdir="y")
        ax.plot(truths[:k], steps[:k], zs=0, zdir="z", color=TRUTH,
                lw=2.2, zorder=10)

        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(0, n)
        ax.set_zlim(0, zmax)
        ax.set_xlabel("position in metres", fontsize=9, color=INK_2, labelpad=10)
        ax.set_ylabel("step", fontsize=9, color=INK_2, labelpad=8)
        ax.tick_params(colors=INK_2, labelsize=7.5)
        ax.view_init(elev=25 + 4 * np.sin(f / 9.0), azim=-68 + 0.85 * f)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor(SURFACE)
            pane.pane.set_edgecolor(GRID)
            pane._axinfo["grid"]["color"] = GRID
        ax.set_zticks([])

        fig.text(0.045, 0.955, "The belief, one bell curve per step",
                 fontsize=14.5, fontweight="bold", color=INK, va="center")
        fig.text(0.045, 0.905,
                 "Row 0 is switch-on. The ridge travels forward with the vehicle "
                 "and grows taller and narrower as the filter becomes certain.",
                 fontsize=9.5, color=INK_2, va="center")
        fig.text(0.045, 0.855, "the orange line is where the vehicle really was",
                 fontsize=9, color=TRUTH, va="center")
        frames.append(grab(fig))
        plt.close(fig)

    return save_gif(frames, out_path, fps, colors=128), len(frames)


# --------------------------------------------------------------------------
# Figure 3, the Kalman gain as a surface
# --------------------------------------------------------------------------
def make_gain_surface(out_path: Path):
    sp = np.linspace(0.2, 9.0, 130)
    ss = np.linspace(0.2, 9.0, 130)
    SP, SS = np.meshgrid(sp, ss)
    K = SP ** 2 / (SP ** 2 + SS ** 2)

    fig = plt.figure(figsize=(9.8, 6.4), dpi=110, facecolor=SURFACE)
    ax = fig.add_axes([0.0, 0.0, 0.86, 0.85], projection="3d", facecolor=SURFACE)
    surf = ax.plot_surface(SP, SS, K, cmap=BLUE_RAMP, linewidth=0, antialiased=True,
                           rstride=2, cstride=2, alpha=0.95,
                           norm=Normalize(0, 1))
    # the same function flattened onto the floor, where values are easy to read off
    ax.contourf(SP, SS, K, levels=np.linspace(0, 1, 11), zdir="z", offset=0,
                cmap=BLUE_RAMP, alpha=0.30, norm=Normalize(0, 1))

    p_ss = steady_state_variance(Q, R)
    op_x, op_y = np.sqrt(p_ss + Q), SIGMA_SENSOR
    op_z = kalman_gain(p_ss + Q, R)
    ax.plot([op_x], [op_y], [op_z], marker="o", ms=11, color=TRUTH,
            markeredgecolor="white", markeredgewidth=1.6, zorder=20)
    ax.plot([op_x, op_x], [op_y, op_y], [0, op_z], color=TRUTH, lw=1.4, ls=(0, (3, 2)))
    # Placed as figure text. Text drawn inside a 3D axes is sorted per artist
    # rather than per fragment, so it disappears behind the surface.
    fig.text(0.035, 0.855,
             f"The orange dot is where this demo settles, K = {op_z:.2f}, so it "
             f"keeps about three quarters of its prediction.",
             fontsize=9.5, color=TRUTH, va="center", fontweight="bold")

    ax.set_xlabel("uncertainty of the prediction, metres", fontsize=9.5,
                  color=INK_2, labelpad=11)
    ax.set_ylabel("uncertainty of the receiver, metres", fontsize=9.5,
                  color=INK_2, labelpad=11)
    ax.set_zlabel("Kalman gain K", fontsize=9.5, color=INK_2, labelpad=8)
    ax.set_zlim(0, 1)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.view_init(elev=25, azim=-56)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(SURFACE)
        pane.pane.set_edgecolor(GRID)
        pane._axinfo["grid"]["color"] = GRID

    fig.text(0.035, 0.955, "Who does the filter believe?", fontsize=14.5,
             fontweight="bold", color=INK, va="center")
    fig.text(0.035, 0.905,
             "K near 1 means the prediction is vague, so follow the receiver. "
             "K near 0 means the receiver is vague, so keep the prediction.",
             fontsize=9.5, color=INK_2, va="center")
    # No colour bar. The height axis already carries K, and a second scale for
    # the same quantity only competes with it.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Figure 4, convergence of the uncertainty from every starting guess
# --------------------------------------------------------------------------
def make_convergence_surface(out_path: Path):
    steps = np.arange(0, 13)
    sigma0 = np.logspace(np.log10(0.3), np.log10(60.0), 90)
    Z = np.zeros((len(sigma0), len(steps)))
    for i, s0 in enumerate(sigma0):
        p = s0 ** 2
        Z[i, 0] = np.sqrt(p)
        for k in steps[1:]:
            p = (p + Q) * R / (p + Q + R)
            Z[i, k] = np.sqrt(p)

    S, T = np.meshgrid(steps, sigma0)
    target = np.sqrt(steady_state_variance(Q, R))

    # Height is plotted on a log scale. The starting guesses span a factor of
    # 200, so on a linear axis the settled value would be a flat line at the
    # bottom and the whole point would be invisible.
    fig = plt.figure(figsize=(9.8, 6.4), dpi=110, facecolor=SURFACE)
    ax = fig.add_axes([0.02, 0.02, 0.94, 0.84], projection="3d", facecolor=SURFACE)
    ax.plot_surface(S, np.log10(T), np.log10(Z), cmap=BLUE_RAMP, linewidth=0.2,
                    edgecolor=(1, 1, 1, 0.18), rstride=3, cstride=1, alpha=0.93)

    ax.set_xlabel("step", fontsize=9.5, color=INK_2, labelpad=8)
    ax.set_ylabel("starting guess, metres", fontsize=9.5, color=INK_2, labelpad=11)
    ax.set_zlabel("uncertainty, metres", fontsize=9.5, color=INK_2, labelpad=9)
    yt = [0.3, 1, 3, 10, 30, 60]
    ax.set_yticks(np.log10(yt))
    ax.set_yticklabels([str(t) for t in yt])
    zt = [0.3, 1, 1.45, 3, 10, 30, 60]
    ax.set_zticks(np.log10(zt))
    ax.set_zticklabels([str(t) for t in zt])
    ax.tick_params(colors=INK_2, labelsize=8)
    # mark the level the whole surface lands on
    for lbl, t in zip(ax.get_zticklabels(), zt):
        if t == 1.45:
            lbl.set_color(TRUTH)
            lbl.set_fontweight("bold")
            lbl.set_fontsize(9.5)
    ax.view_init(elev=24, azim=-61)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(SURFACE)
        pane.pane.set_edgecolor(GRID)
        pane._axinfo["grid"]["color"] = GRID

    fig.text(0.035, 0.955, "The uncertainty schedule is fixed in advance",
             fontsize=14.5, fontweight="bold", color=INK, va="center")
    fig.text(0.035, 0.905,
             "Ninety starting guesses, from 0.3 m to 60 m, on a log height scale. "
             "Every one of them falls onto the same plateau within a few steps.",
             fontsize=9.5, color=INK_2, va="center")
    fig.text(0.035, 0.855,
             f"That plateau is {target:.2f} m, marked in orange on the height axis. "
             f"No measurement was involved in deciding it.",
             fontsize=9.5, color=TRUTH, va="center", fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Figure 5, the whole run as one still image
# --------------------------------------------------------------------------
def make_tracking_summary(phases: list[Phase], out_path: Path):
    posts = [p for p in phases if p.kind == "update"]
    k = np.array([p.step for p in posts])
    truth = np.array([p.truth for p in posts])
    meas = np.array([p.measurement for p in posts])
    mu = np.array([p.after.mean for p in posts])
    sd = np.array([p.after.std for p in posts])
    gain = np.array([p.gain for p in posts])
    target = np.sqrt(steady_state_variance(Q, R))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.2, 8.4), dpi=110,
                                        facecolor=SURFACE,
                                        height_ratios=[1.7, 1.0, 0.8],
                                        sharex=True)
    for ax in (ax1, ax2, ax3):
        style_axes(ax)
    ax3.set_xticks(range(0, N_STEPS + 1, 2))

    ax1.fill_between(k, mu - 2 * sd, mu + 2 * sd, color=BELIEF, alpha=0.18,
                     label="range the filter considers likely")
    ax1.plot(k, meas, ls="none", marker="o", ms=6, color=SENSOR,
             markeredgecolor=SURFACE, markeredgewidth=1.0, label="receiver reading")
    ax1.plot(k, truth, color=TRUTH, lw=2.3, label="where the vehicle really is")
    ax1.plot(k, mu, color=BELIEF, lw=2.1, label="filter estimate")
    ax1.set_ylabel("position in metres", fontsize=10, color=INK_2)
    ax1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2, ncol=2)
    ax1.set_title("The filter is closer to the truth than the receiver it is fed",
                  fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=12)

    ax2.plot(k, sd, color=BELIEF, lw=2.2, marker="o", ms=4.5,
             markeredgecolor=SURFACE)
    ax2.axhline(SIGMA_SENSOR, color=SENSOR_DK, lw=1.4, ls=(0, (2, 2)))
    ax2.axhline(target, color=TRUTH, lw=1.6, ls=(0, (4, 3)))
    ax2.text(k[0] + 0.1, SIGMA_SENSOR + 0.18,
             f"the receiver on its own is {SIGMA_SENSOR:.1f} m",
             fontsize=9, color=SENSOR_DK, ha="left")
    ax2.text(k[-1], target + 0.22, f"the filter settles at {target:.2f} m",
             fontsize=9, color=TRUTH, ha="right")
    ax2.set_ylim(0, 4.2)
    ax2.set_ylabel("uncertainty of the\nestimate, metres", fontsize=10, color=INK_2)

    ax3.plot(k, gain, color=BELIEF, lw=2.0, marker="o", ms=4.0,
             markeredgecolor=SURFACE)
    ax3.axhline(gain[-1], color=TRUTH, lw=1.4, ls=(0, (4, 3)))
    ax3.text(k[-1], gain[-1] + 0.07, f"settles at {gain[-1]:.2f}", fontsize=9,
             color=TRUTH, ha="right")
    ax3.text(5.2, 0.85,
             "high K means believe the receiver, low K means believe the prediction",
             fontsize=8.5, color=INK_3, ha="left", style="italic")
    ax3.set_ylim(0, 1.0)
    ax3.set_xlabel("step", fontsize=10, color=INK_2)
    ax3.set_ylabel("Kalman gain K", fontsize=10, color=INK_2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
def print_trace(phases: list[Phase]):
    print(f"commanded step {COMMAND} m,  wheel noise {SIGMA_PROCESS} m,  "
          f"receiver noise {SIGMA_SENSOR} m,  initial uncertainty {SIGMA_INITIAL} m\n")
    print("step        event      estimate   sigma     truth     error     K")
    for p in phases:
        if p.kind == "start":
            print(f"  0    switched on    {p.after.mean:7.2f}  {p.after.std:6.2f}   "
                  f"{p.truth:7.2f}   {p.after.mean - p.truth:+6.2f}      -")
            continue
        g = f"{p.gain:.3f}" if p.gain is not None else "    -"
        label = "predict" if p.kind == "predict" else "update "
        print(f" {p.step:2d}       {label}     {p.after.mean:7.2f}  {p.after.std:6.2f}   "
              f"{p.truth:7.2f}   {p.after.mean - p.truth:+6.2f}  {g}")


def accuracy_study(n_runs=400, warmup=8) -> tuple[float, float]:
    """How much better is the filter than the receiver it is fed?"""
    rng = np.random.default_rng(404)
    motion = Gaussian(COMMAND, Q)
    ef, eg = [], []
    for _ in range(n_runs):
        x, belief = START, Gaussian(START, SIGMA_INITIAL ** 2)
        for k in range(N_STEPS + 12):
            x += COMMAND + rng.normal(0, SIGMA_PROCESS)
            z = x + rng.normal(0, SIGMA_SENSOR)
            belief, _ = update(predict(belief, motion), Gaussian(z, R))
            if k >= warmup:
                ef.append(belief.mean - x)
                eg.append(z - x)
    return (float(np.sqrt(np.mean(np.square(eg)))),
            float(np.sqrt(np.mean(np.square(ef)))))


def main():
    ap = argparse.ArgumentParser(description="Day 2 Kalman filter demo")
    ap.add_argument("--no-gif", action="store_true", help="skip the two animations")
    args = ap.parse_args()

    phases = simulate()
    print_trace(phases)

    worst = validate_against_filterpy(phases)
    print(f"\nlargest disagreement with filterpy over the whole run  {worst:.3e}")

    p_ss = steady_state_variance(Q, R)
    print(f"predicted steady state sigma  {np.sqrt(p_ss):.4f} m      "
          f"gain {steady_state_gain(Q, R):.4f}")
    print(f"reached by step {N_STEPS}         "
          f"{phases[-1].after.std:.4f} m      gain {phases[-1].gain:.4f}")

    raw, filt = accuracy_study()
    print(f"\nover 400 runs   receiver alone {raw:.3f} m RMS   "
          f"filtered {filt:.3f} m RMS   improvement {raw / filt:.2f}x")

    media = Path(__file__).parent / "media"
    print()
    print("wrote", make_tracking_summary(phases, media / "tracking_summary.png"))
    print("wrote", make_gain_surface(media / "kalman_gain_surface.png"))
    print("wrote", make_convergence_surface(media / "variance_convergence_3d.png"))
    if not args.no_gif:
        g, n = make_main_gif(phases, media / "kalman_filter_1d.gif")
        print(f"wrote {g}  ({n} frames, {g.stat().st_size / 1e6:.1f} MB)")
        g, n = make_ridge_gif(phases, media / "belief_ridge_3d.gif")
        print(f"wrote {g}  ({n} frames, {g.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
