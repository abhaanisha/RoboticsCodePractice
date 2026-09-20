"""
Day 3 demo - tracking position and velocity when only position is measured.

A vehicle drives along a straight road at a roughly constant speed. A
satellite receiver reports its position and nothing else. There is no
speedometer anywhere in the system. The filter is asked for both the position
and the velocity.

Outputs, into ./media
    kalman_filter_nd.gif       the main animation
    belief_hill_3d.gif         the two variable Gaussian as a rotating surface
    covariance_tower_3d.png    the uncertainty ellipse at every step, stacked
    shear_and_squeeze.png      how a position reading shrinks velocity error
    tracking_summary.png       the whole run as one still image

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from PIL import Image
from filterpy.common import Q_discrete_white_noise

from kalman_filter_nd import (GaussianState, predict, update, steady_state,
                              ellipse_coverage)

# --------------------------------------------------------------------------
# The scenario
# --------------------------------------------------------------------------
DT = 1.0
N_STEPS = 25
TRUE_SPEED = 2.0        # metres per second
SIGMA_ACCEL = 0.25      # metres per second squared, how steady the speed is
SIGMA_SENSOR = 3.0      # metres, the receiver, same as day 2
SIGMA_P0 = 8.0          # metres, initial doubt about position
SIGMA_V0 = 3.0          # metres per second, initial doubt about speed
SEED = 3

F = np.array([[1.0, DT], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = Q_discrete_white_noise(dim=2, dt=DT, var=SIGMA_ACCEL ** 2)
R = np.array([[SIGMA_SENSOR ** 2]])

# --------------------------------------------------------------------------
# Palette, unchanged since day 1
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e1"
BELIEF = "#2a78d6"
BELIEF_LT = "#cde2fb"
TRUTH = "#eb6834"
SENSOR = "#1baf7a"
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
    kind: str                 # "start" | "predict" | "update"
    step: int
    before: GaussianState
    after: GaussianState
    truth: np.ndarray
    measurement: float | None = None
    gain: np.ndarray | None = None


def simulate() -> list[Phase]:
    rng = np.random.default_rng(SEED)
    x = np.array([0.0, TRUE_SPEED])
    state = GaussianState([0.0, 0.0], np.diag([SIGMA_P0 ** 2, SIGMA_V0 ** 2]))
    phases = [Phase("start", 0, state, state, x.copy())]

    for k in range(1, N_STEPS + 1):
        # a random nudge to the acceleration, which is what Q models
        a = rng.normal(0.0, SIGMA_ACCEL)
        x = F @ x + np.array([0.5 * DT * DT, DT]) * a

        prior = predict(state, F, Q)
        phases.append(Phase("predict", k, state, prior, x.copy()))

        z = x[0] + rng.normal(0.0, SIGMA_SENSOR)
        res = update(prior, [z], H, R)
        phases.append(Phase("update", k, prior, res.state, x.copy(),
                            measurement=z, gain=res.gain))
        state = res.state

    return phases


def validate_against_filterpy(phases: list[Phase]) -> float:
    from filterpy.kalman import KalmanFilter
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.0], [0.0]])
    kf.P = np.diag([SIGMA_P0 ** 2, SIGMA_V0 ** 2])
    kf.F, kf.H, kf.Q, kf.R = F, H, Q, R

    worst = 0.0
    for p in phases:
        if p.kind == "predict":
            kf.predict()
        elif p.kind == "update":
            kf.update(np.array([[p.measurement]]))
        else:
            continue
        worst = max(worst,
                    np.abs(kf.x.ravel() - p.after.mean).max(),
                    np.abs(kf.P - p.after.cov).max())
    return float(worst)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def blend(a: GaussianState, b: GaussianState, t: float) -> GaussianState:
    return GaussianState(a.mean + t * (b.mean - a.mean),
                         a.cov + t * (b.cov - a.cov))


def save_gif(frames, out_path: Path, fps: int, colors: int = 192):
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
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)


def style_3d(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(SURFACE)
        pane.pane.set_edgecolor(GRID)
        pane._axinfo["grid"]["color"] = GRID
    ax.tick_params(colors=INK_2, labelsize=7.5)


def error_state(state: GaussianState, truth: np.ndarray) -> GaussianState:
    """The belief expressed as an error relative to the true state."""
    return GaussianState(state.mean - truth, state.cov)


# --------------------------------------------------------------------------
# Figure 1, the main animation
# --------------------------------------------------------------------------
def make_main_gif(phases, out_path: Path, fps=14, ease=4, hold=2, max_step=12):
    ph = [p for p in phases if p.step <= max_step]
    ks = [p.step for p in ph]
    tp = [p.truth[0] for p in ph]

    fig = plt.figure(figsize=(12.2, 6.7), dpi=76, facecolor=SURFACE)
    ax_ph = fig.add_axes([0.055, 0.135, 0.36, 0.63])
    ax_p = fig.add_axes([0.545, 0.475, 0.425, 0.29])
    ax_v = fig.add_axes([0.545, 0.135, 0.425, 0.245])
    for a in (ax_ph, ax_p, ax_v):
        style_axes(a)

    fig.text(0.055, 0.945, "Kalman Filter with Two Variables", fontsize=15.5,
             fontweight="bold", color=INK, va="center")
    fig.text(0.968, 0.945, "Day 3 / Localization", fontsize=9.5, color=INK_3,
             va="center", ha="right")
    badge = fig.text(0.055, 0.872, " PREDICT ", fontsize=9.5, fontweight="bold",
                     color=TRUTH, va="center", family="monospace",
                     bbox=dict(boxstyle="round,pad=0.42", fc=PREDICT_BG, ec="none"))
    caption = fig.text(0.163, 0.872, "", fontsize=10.5, color=INK_2, va="center")

    # ---- phase space, drawn as error from the true state ----
    ax_ph.set_xlim(-17, 17)
    ax_ph.set_ylim(-6.8, 6.8)
    ax_ph.axhline(0, color=GRID, lw=1.0, zorder=1)
    ax_ph.axvline(0, color=GRID, lw=1.0, zorder=1)
    ax_ph.set_xlabel("position error, metres", fontsize=9.5, color=INK_2)
    ax_ph.set_ylabel("velocity error, metres per second", fontsize=9.5, color=INK_2)
    ax_ph.set_title("The belief, seen as a cloud over both variables",
                    fontsize=10.5, color=INK, loc="left", pad=8)
    band = ax_ph.axvspan(0, 0, color=SENSOR, alpha=0.16, zorder=2)
    ell2 = Polygon(np.zeros((3, 2)), closed=True, facecolor=BELIEF, alpha=0.20,
                   edgecolor=BELIEF, lw=1.8, zorder=4)
    ell1 = Polygon(np.zeros((3, 2)), closed=True, facecolor=BELIEF, alpha=0.30,
                   edgecolor="none", zorder=5)
    ax_ph.add_patch(ell2)
    ax_ph.add_patch(ell1)
    centre, = ax_ph.plot([], [], marker="o", ms=6, color=BELIEF,
                         markeredgecolor=SURFACE, markeredgewidth=1.2, ls="none",
                         zorder=6)
    ax_ph.plot([0], [0], marker="X", ms=13, color=TRUTH, markeredgecolor=SURFACE,
               markeredgewidth=1.4, ls="none", zorder=7)
    ax_ph.text(0.6, 0.35, "the truth", fontsize=8.5, color=TRUTH)
    corr_txt = ax_ph.text(-16.2, 5.9, "", fontsize=9, color=INK_2)

    # ---- position against time ----
    ax_p.set_xlim(-0.5, max_step + 0.5)
    ax_p.set_ylim(min(tp) - 9, max(tp) + 10)
    ax_p.set_xticks(range(0, max_step + 1, 2))
    ax_p.set_ylabel("position, m", fontsize=9.5, color=INK_2)
    ax_p.set_title("Position, which the receiver does report",
                   fontsize=10.5, color=INK, loc="left", pad=8)
    pband = ax_p.fill_between([], [], [], color=BELIEF, alpha=0.18, zorder=2)
    p_true, = ax_p.plot([], [], color=TRUTH, lw=2.2, zorder=5)
    p_est, = ax_p.plot([], [], color=BELIEF, lw=2.0, zorder=4)
    p_meas, = ax_p.plot([], [], ls="none", marker="o", ms=7, color=SENSOR,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=6)

    # ---- velocity against time ----
    ax_v.set_xlim(-0.5, max_step + 0.5)
    ax_v.set_ylim(-4.5, 7.0)
    ax_v.set_xticks(range(0, max_step + 1, 2))
    ax_v.set_xlabel("step", fontsize=9.5, color=INK_2)
    ax_v.set_ylabel("velocity, m/s", fontsize=9.5, color=INK_2)
    ax_v.set_title("Velocity, which nothing in the system ever measures",
                   fontsize=10.5, color=TRUTH, loc="left", pad=8)
    vband = ax_v.fill_between([], [], [], color=BELIEF, alpha=0.18, zorder=2)
    v_true, = ax_v.plot([], [], color=TRUTH, lw=2.2, zorder=5)
    v_est, = ax_v.plot([], [], color=BELIEF, lw=2.0, zorder=4)

    fig.legend(handles=[
        Line2D([], [], color=TRUTH, lw=2.2, label="the truth"),
        Line2D([], [], ls="none", marker="o", ms=7, color=SENSOR,
               markeredgecolor=SURFACE, label="receiver reading, position only"),
        Line2D([], [], color=BELIEF, lw=2.2, label="the filter estimate"),
        Line2D([], [], color=BELIEF, lw=7, alpha=0.18, label="two sigma range"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.085), frameon=False, ncol=4,
        fontsize=8.5, labelcolor=INK_2, handletextpad=0.5, columnspacing=2.0)
    footer = fig.text(0.5, 0.022, "", fontsize=9.5, color=INK_2, va="center",
                      ha="center", family="monospace")

    hk, hpt, hpe, hplo, hphi = [], [], [], [], []
    hvt, hve, hvlo, hvhi = [], [], [], []
    mk, mz = [], []
    frames = []

    def draw(phase, st, show_meas):
        nonlocal band, pband, vband
        err = error_state(st, phase.truth)
        for poly, n in ((ell2, 2.0), (ell1, 1.0)):
            ex, ey = err.ellipse(n)
            poly.set_xy(np.column_stack([ex, ey]))
        centre.set_data([err.mean[0]], [err.mean[1]])
        rho = err.correlation[0, 1]
        corr_txt.set_text(f"correlation between the two errors  {rho:+.2f}")

        band.remove()
        if show_meas and phase.measurement is not None:
            d = phase.measurement - phase.truth[0]
            band = ax_ph.axvspan(d - SIGMA_SENSOR, d + SIGMA_SENSOR,
                                 color=SENSOR, alpha=0.16, zorder=2)
        else:
            band = ax_ph.axvspan(0, 0, color=SENSOR, alpha=0.0, zorder=2)

        p_true.set_data(hk, hpt)
        p_est.set_data(hk, hpe)
        p_meas.set_data(mk, mz)
        v_true.set_data(hk, hvt)
        v_est.set_data(hk, hve)
        pband.remove()
        vband.remove()
        pband = ax_p.fill_between(hk, hplo, hphi, color=BELIEF, alpha=0.18, zorder=2)
        vband = ax_v.fill_between(hk, hvlo, hvhi, color=BELIEF, alpha=0.18, zorder=2)

        footer.set_text(
            f"position {st.mean[0]:6.2f} +/- {st.std[0]:4.2f} m     "
            f"velocity {st.mean[1]:5.2f} +/- {st.std[1]:4.2f} m/s     "
            f"true velocity {phase.truth[1]:5.2f} m/s")

    for idx, phase in enumerate(ph):
        if phase.kind == "start":
            bt, bc, bg = " START   ", INK_2, "#efeeea"
            cap = ("Position unknown to about 8 m, velocity a complete guess of zero. "
                   "The cloud is wide and upright.")
        elif phase.kind == "predict":
            bt, bc, bg = " PREDICT ", TRUTH, PREDICT_BG
            cap = (f"step {phase.step}. Roll the state forward one second. "
                   f"The cloud tilts, because position depends on velocity.")
        else:
            bt, bc, bg = " UPDATE  ", BELIEF, UPDATE_BG
            cap = (f"step {phase.step}. A position reading arrives, the green band. "
                   f"It is vertical, yet the cloud shrinks vertically too.")
        badge.set_text(bt)
        badge.set_color(bc)
        badge.get_bbox_patch().set_facecolor(bg)
        caption.set_text(cap)

        if phase.kind == "update":
            hk.append(phase.step)
            hpt.append(phase.truth[0])
            hvt.append(phase.truth[1])
            hpe.append(phase.after.mean[0])
            hve.append(phase.after.mean[1])
            hplo.append(phase.after.mean[0] - 2 * phase.after.std[0])
            hphi.append(phase.after.mean[0] + 2 * phase.after.std[0])
            hvlo.append(phase.after.mean[1] - 2 * phase.after.std[1])
            hvhi.append(phase.after.mean[1] + 2 * phase.after.std[1])
            mk.append(phase.step)
            mz.append(phase.measurement)

        n_ease = 1 if phase.kind == "start" else ease
        n_hold = hold + (8 if phase.kind == "start" else 0)
        if idx == len(ph) - 1:
            n_hold += 20
        for f in range(n_ease):
            t = smoothstep((f + 1) / n_ease)
            draw(phase, blend(phase.before, phase.after, t), phase.kind == "update")
            frames.append(grab(fig))
        frames.extend([frames[-1]] * n_hold)

    plt.close(fig)
    return save_gif(frames, out_path, fps), len(frames)


# --------------------------------------------------------------------------
# Figure 2, the belief as a rotating hill
# --------------------------------------------------------------------------
def make_hill_gif(phases, out_path: Path, fps=12, ease=5,
                  first_step=2, max_step=8):
    # Every hill is drawn to the same peak height. The true peak rises by a
    # factor of thirty five across the run, so on a common height scale the
    # early hills would be invisible pancakes. Scaling them removes the size
    # information, which the floor ellipses and figure 3.5 carry instead, and
    # leaves the shape, which is what this picture is for.
    ph = [p for p in phases if first_step <= p.step <= max_step]
    gx = np.linspace(-14, 14, 100)
    gy = np.linspace(-6.0, 6.0, 100)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.dstack([GX, GY])


    frames, f_i = [], 0
    for phase in ph:
        if phase.kind == "predict":
            cap = (f"step {phase.step}. PREDICT leans the hill over. "
                   f"No measurement was involved.")
            col = TRUTH
        else:
            cap = (f"step {phase.step}. UPDATE pulls the footprint in, "
                   f"along both axes at once.")
            col = BELIEF
        for f in range(ease):
            t = smoothstep((f + 1) / ease)
            st = error_state(blend(phase.before, phase.after, t), phase.truth)
            Z = st.pdf(pts)
            Z = Z / Z.max()

            fig = plt.figure(figsize=(9.6, 6.3), dpi=80, facecolor=SURFACE)
            ax = fig.add_axes([-0.01, 0.005, 0.90, 0.80], projection="3d",
                              facecolor=SURFACE)
            ax.plot_surface(GX, GY, Z, cmap=BLUE_RAMP, linewidth=0.15,
                            edgecolor=(1, 1, 1, 0.16), rstride=2, cstride=2,
                            alpha=0.95, vmin=0, vmax=1.0)
            ax.set_xlim(gx[0], gx[-1])
            ax.set_ylim(gy[0], gy[-1])
            ax.set_zlim(0, 1.06)
            ax.set_xlabel("position error, m", fontsize=9, color=INK_2, labelpad=9)
            ax.set_ylabel("velocity error, m/s", fontsize=9, color=INK_2, labelpad=9)
            ax.set_zticks([])
            style_3d(ax)
            ax.view_init(elev=28 + 5 * np.sin(f_i / 11.0), azim=-62 + 0.8 * f_i)

            # The footprint as a flat inset rather than as rings on the 3D
            # floor. Matplotlib sorts whole artists rather than fragments, so
            # anything drawn on that floor spends most of the rotation hidden
            # behind the skirt of the hill.
            ins = fig.add_axes([0.745, 0.415, 0.225, 0.30], facecolor=SURFACE)
            style_axes(ins)
            ins.axhline(0, color=GRID, lw=0.9)
            ins.axvline(0, color=GRID, lw=0.9)
            for n_std, al in ((2.0, 0.18), (1.0, 0.30)):
                ex, ey = st.ellipse(n_std)
                ins.fill(ex, ey, color=BELIEF, alpha=al)
            ex, ey = st.ellipse(2.0)
            ins.plot(ex, ey, color=BELIEF, lw=1.6)
            ins.plot([0], [0], marker="X", ms=9, color=TRUTH,
                     markeredgecolor=SURFACE, markeredgewidth=1.2, ls="none")
            ins.set_xlim(-11, 11)
            ins.set_ylim(-4.6, 4.6)
            ins.set_xticks([])
            ins.set_yticks([])
            ins.set_title("footprint, seen from above", fontsize=8.5, color=INK_2,
                          pad=5)

            fig.text(0.040, 0.962, "One hill over two variables", fontsize=14.5,
                     fontweight="bold", color=INK, va="center")
            fig.text(0.040, 0.920,
                     "Heights are scaled to a common peak, so the shapes can be "
                     "compared. The orange cross is zero error.",
                     fontsize=9, color=INK_2, va="center")
            fig.text(0.040, 0.878, cap, fontsize=10, color=col, va="center",
                     fontweight="bold")
            frames.append(grab(fig))
            plt.close(fig)
            f_i += 1
        frames.extend([frames[-1]] * 2)

    frames.extend([frames[-1]] * 12)
    return save_gif(frames, out_path, fps, colors=200), len(frames)


# --------------------------------------------------------------------------
# Figure 3, every uncertainty ellipse stacked along time
# --------------------------------------------------------------------------
def make_tower(phases, out_path: Path):
    posts = [p for p in phases if p.kind in ("start", "update") and p.step <= 16]
    fig = plt.figure(figsize=(9.8, 6.6), dpi=110, facecolor=SURFACE)
    ax = fig.add_axes([0.02, 0.02, 0.94, 0.82], projection="3d", facecolor=SURFACE)

    n = len(posts)
    for i, p in enumerate(posts):
        err = error_state(p.after, p.truth)
        ex, ey = err.ellipse(2.0)
        c = BLUE_RAMP(0.12 + 0.88 * i / (n - 1))
        ax.plot(ex, np.full_like(ex, p.step), ey, color=c, lw=1.7,
                alpha=0.95 if i % 2 == 0 else 0.55)
    ax.plot([0, 0], [0, posts[-1].step], [0, 0], color=TRUTH, lw=2.4, zorder=30)

    ax.set_xlim(-17, 17)
    ax.set_zlim(-6.5, 6.5)
    ax.set_ylim(0, posts[-1].step)
    ax.set_xlabel("position error, metres", fontsize=9.5, color=INK_2, labelpad=9)
    ax.set_ylabel("step", fontsize=9.5, color=INK_2, labelpad=7)
    ax.set_zlabel("velocity error, m/s", fontsize=9.5, color=INK_2, labelpad=6)
    style_3d(ax)
    ax.view_init(elev=14, azim=-72)

    fig.text(0.035, 0.955, "Every uncertainty ellipse, stacked along time",
             fontsize=14.5, fontweight="bold", color=INK, va="center")
    fig.text(0.035, 0.908,
             "Each ring is the two sigma ellipse at one step. They shrink, and "
             "they lean over as position error and velocity error become linked.",
             fontsize=9.5, color=INK_2, va="center")
    fig.text(0.035, 0.862,
             "The orange line is zero error. A well behaved filter keeps it inside "
             "the rings about 86 percent of the time.",
             fontsize=9.5, color=TRUTH, va="center", fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Figure 4, the mechanism in three panels
# --------------------------------------------------------------------------
def make_shear_squeeze(out_path: Path):
    s0 = GaussianState([0.0, 0.0], np.diag([9.0, 4.0]))
    s1 = predict(s0, F, np.zeros((2, 2)))
    res = update(s1, [0.0], H, np.array([[9.0]]))
    s2 = res.state

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.5), dpi=110,
                             facecolor=SURFACE, sharex=True, sharey=True)
    titles = ["a.  before", "b.  after PREDICT", "c.  after UPDATE"]
    notes = [
        "Position and velocity errors are\nunrelated. The ellipse is upright.",
        "Rolling forward tilts it. A fast\nrobot overshoots, so the two errors\n"
        "are now linked. Nothing was measured.",
        "The green band is a position reading.\nIt constrains position only, yet the\n"
        "tilt drags the velocity down with it.",
    ]
    for ax, st, title, note in zip(axes, (s0, s1, s2), titles, notes):
        style_axes(ax)
        ax.axhline(0, color=GRID, lw=1.0)
        ax.axvline(0, color=GRID, lw=1.0)
        for n, al in ((2.0, 0.18), (1.0, 0.30)):
            ex, ey = st.ellipse(n)
            ax.fill(ex, ey, color=BELIEF, alpha=al, zorder=3)
        ex, ey = st.ellipse(2.0)
        ax.plot(ex, ey, color=BELIEF, lw=1.8, zorder=4)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK,
                     loc="left", pad=10)
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=8.5,
                color=INK_2, va="bottom")
        ax.set_xlabel("position error, metres", fontsize=9.5, color=INK_2)
        sv = st.std[1]
        ax.annotate("", xy=(-9.4, sv), xytext=(-9.4, -sv),
                    arrowprops=dict(arrowstyle="<->", color=TRUTH, lw=1.7))
        ax.text(-8.9, 0, f"velocity\n{sv:.2f} m/s", fontsize=9, color=TRUTH,
                va="center", fontweight="bold")

    axes[2].axvspan(-3, 3, color=SENSOR, alpha=0.16, zorder=1)
    axes[2].text(0, 4.5, "position reading", fontsize=8.5, color=SENSOR_DK,
                 ha="center", fontweight="bold")
    axes[0].set_ylabel("velocity error, metres per second", fontsize=9.5, color=INK_2)
    axes[0].set_xlim(-11, 11)
    axes[0].set_ylim(-5.2, 5.2)

    fig.suptitle("A reading about position alone makes the velocity estimate better",
                 fontsize=14, fontweight="bold", color=INK, x=0.012, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Figure 5, the run as one still image
# --------------------------------------------------------------------------
def make_summary(phases, out_path: Path):
    posts = [p for p in phases if p.kind == "update"]
    k = np.array([p.step for p in posts])
    tp = np.array([p.truth[0] for p in posts])
    tv = np.array([p.truth[1] for p in posts])
    mz = np.array([p.measurement for p in posts])
    mp = np.array([p.after.mean[0] for p in posts])
    mv = np.array([p.after.mean[1] for p in posts])
    sp = np.array([p.after.std[0] for p in posts])
    sv = np.array([p.after.std[1] for p in posts])
    rho = np.array([p.after.correlation[0, 1] for p in posts])
    _, P_ss, _ = steady_state(F, H, Q, R)

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10.2, 8.6), dpi=110,
                                     facecolor=SURFACE, sharex=True,
                                     height_ratios=[1.35, 1.35, 0.85])
    for a in (a1, a2, a3):
        style_axes(a)
    a3.set_xticks(range(0, N_STEPS + 1, 2))

    a1.fill_between(k, mp - 2 * sp, mp + 2 * sp, color=BELIEF, alpha=0.18,
                    label="two sigma range")
    a1.plot(k, mz, ls="none", marker="o", ms=6, color=SENSOR,
            markeredgecolor=SURFACE, markeredgewidth=1.0, label="receiver reading")
    a1.plot(k, tp, color=TRUTH, lw=2.3, label="the truth")
    a1.plot(k, mp, color=BELIEF, lw=2.0, label="estimate")
    a1.set_ylabel("position, metres", fontsize=10, color=INK_2)
    a1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2, ncol=2)
    a1.set_title("Position is measured, so tracking it is no surprise",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10)

    a2.fill_between(k, mv - 2 * sv, mv + 2 * sv, color=BELIEF, alpha=0.18)
    a2.plot(k, tv, color=TRUTH, lw=2.3)
    a2.plot(k, mv, color=BELIEF, lw=2.0)
    a2.axhline(0, color=INK_3, lw=0.9, ls=(0, (3, 3)))
    a2.set_ylabel("velocity, metres\nper second", fontsize=10, color=INK_2)
    a2.set_ylim(-7.0, 7.5)
    a2.text(0.4, -6.3, "the estimate starts at zero, a pure guess, so the first "
                       "few two sigma bands are enormous",
            fontsize=8.5, color=INK_3, style="italic")
    a2.set_title("Velocity is never measured, and is tracked anyway",
                 fontsize=12.5, fontweight="bold", color=TRUTH, loc="left", pad=10)
    a2.text(k[-1], tv[-1] + 1.6, f"settles to +/- {np.sqrt(P_ss[1,1]):.2f} m/s",
            fontsize=9, color=TRUTH, ha="right", fontweight="bold")

    a3.plot(k, rho, color=BELIEF, lw=2.1, marker="o", ms=4,
            markeredgecolor=SURFACE)
    a3.axhline(P_ss[0, 1] / np.sqrt(P_ss[0, 0] * P_ss[1, 1]), color=TRUTH,
               lw=1.5, ls=(0, (4, 3)))
    a3.set_ylim(0, 1)
    a3.set_xlabel("step", fontsize=10, color=INK_2)
    a3.set_ylabel("correlation between\nthe two errors", fontsize=10, color=INK_2)
    a3.text(k[-1], 0.80, "this number is the reason the row above works",
            fontsize=9, color=TRUTH, ha="right", fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
def print_trace(phases):
    print(f"model  constant velocity,  dt {DT} s,  acceleration noise "
          f"{SIGMA_ACCEL} m/s^2,  receiver noise {SIGMA_SENSOR} m")
    print(f"start  position 0 +/- {SIGMA_P0} m,  velocity 0 +/- {SIGMA_V0} m/s  "
          f"(the true speed is {TRUE_SPEED} m/s)\n")
    print("step     event    est pos   sigma    est vel   sigma   true vel   corr")
    for p in phases:
        if p.kind == "predict" and p.step % 4 and p.step > 4:
            continue
        if p.kind == "update" and p.step % 4 and p.step > 4:
            continue
        lab = {"start": "start  ", "predict": "predict", "update": "update "}[p.kind]
        s = p.after
        print(f" {p.step:2d}    {lab}   {s.mean[0]:7.2f}  {s.std[0]:6.2f}   "
              f"{s.mean[1]:6.2f}  {s.std[1]:6.2f}    {p.truth[1]:5.2f}   "
              f"{s.correlation[0,1]:+.3f}")


def accuracy_study(n_runs=400, warmup=12):
    rng = np.random.default_rng(909)
    pe, ve, ze = [], [], []
    for _ in range(n_runs):
        x = np.array([0.0, TRUE_SPEED])
        st = GaussianState([0.0, 0.0], np.diag([SIGMA_P0 ** 2, SIGMA_V0 ** 2]))
        for k in range(N_STEPS + 8):
            a = rng.normal(0.0, SIGMA_ACCEL)
            x = F @ x + np.array([0.5 * DT * DT, DT]) * a
            z = x[0] + rng.normal(0.0, SIGMA_SENSOR)
            st = update(predict(st, F, Q), [z], H, R).state
            if k >= warmup:
                pe.append(st.mean[0] - x[0])
                ve.append(st.mean[1] - x[1])
                ze.append(z - x[0])
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    return rms(ze), rms(pe), rms(ve)


def main():
    ap = argparse.ArgumentParser(description="Day 3 multivariate Kalman demo")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    phases = simulate()
    print_trace(phases)

    print(f"\nlargest disagreement with filterpy   "
          f"{validate_against_filterpy(phases):.3e}")

    _, P_ss, K_ss = steady_state(F, H, Q, R)
    print(f"closed form steady state (scipy solve_discrete_are)")
    print(f"   position {np.sqrt(P_ss[0,0]):.4f} m      velocity "
          f"{np.sqrt(P_ss[1,1]):.4f} m/s      correlation "
          f"{P_ss[0,1]/np.sqrt(P_ss[0,0]*P_ss[1,1]):+.4f}")
    last = phases[-1].after
    print(f"   reached  {last.std[0]:.4f} m      {last.std[1]:.4f} m/s      "
          f"{last.correlation[0,1]:+.4f}")
    print(f"   steady state gain K = {K_ss.ravel()}")
    print(f"a two sigma ellipse in two dimensions covers "
          f"{100*ellipse_coverage(2.0, 2):.1f} percent, not 95.4")

    raw, pos, vel = accuracy_study()
    print(f"\nover 400 runs after warm-up")
    print(f"   receiver position  {raw:.3f} m RMS")
    print(f"   filtered position  {pos:.3f} m RMS   ({raw/pos:.2f}x better)")
    print(f"   velocity           {vel:.3f} m/s RMS  from no velocity sensor at all")

    media = Path(__file__).parent / "media"
    print()
    print("wrote", make_shear_squeeze(media / "shear_and_squeeze.png"))
    print("wrote", make_summary(phases, media / "tracking_summary.png"))
    print("wrote", make_tower(phases, media / "covariance_tower_3d.png"))
    if not args.no_gif:
        g, n = make_main_gif(phases, media / "kalman_filter_nd.gif")
        print(f"wrote {g}  ({n} frames, {g.stat().st_size/1e6:.1f} MB)")
        g, n = make_hill_gif(phases, media / "belief_hill_3d.gif")
        print(f"wrote {g}  ({n} frames, {g.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
