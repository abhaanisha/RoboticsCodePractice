"""
Day 1 demo - watch a histogram filter figure out where it is.

Simulates a robot in a 20-cell cyclic corridor. Some cells have doors. The
robot can only tell "door" / "no door", and its wheels sometimes slip. It
starts with no idea where it is and has to work it out.

Outputs (into ./media):
    histogram_filter_1d.gif   - the animation
    belief_evolution.png      - a single-image summary of the same run

Run:  python run_demo.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import histogram_filter as hf

# --------------------------------------------------------------------------
# The scenario
# --------------------------------------------------------------------------
N_CELLS = 20
DOOR_CELLS = [6, 7, 8, 11, 13, 14]
START_CELL = 3
COMMAND = 1                    # drive one cell to the right each step
N_STEPS = 10
P_SENSOR_CORRECT = 0.92        # the door detector is right 92% of the time
P_MOVE_EXACT = 0.90            # the wheels obey the command 90% of the time
SEED = 3749                    # chosen so the run contains one sensor glitch
                               # AND one wheel slip - see README

# --------------------------------------------------------------------------
# Palette (validated CVD-safe; see README)
# --------------------------------------------------------------------------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_3     = "#8a8983"
GRID      = "#e6e5e1"
BELIEF    = "#2a78d6"   # slot 1 - the robot belief
TRUTH     = "#eb6834"   # slot 2 - ground truth (known to us, not to the robot)
DOOR      = "#1baf7a"   # slot 3 - landmarks on the map
DOOR_DK   = "#12805a"
WALL      = "#f0efec"
WALL_EDGE = "#dcdad5"
SENSE_BG  = "#e8f1fd"
MOVE_BG   = "#fdeee7"


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
@dataclass
class Phase:
    """One animated beat: either a SENSE update or a MOVE update."""
    kind: str                 # "wake" | "sense" | "move"
    step: int
    belief_from: np.ndarray
    belief_to: np.ndarray
    pos_from: int
    pos_to: int
    reading: bool | None = None
    glitch: bool = False      # sensor reported the wrong thing
    slip: bool = False        # wheels did not obey the command


def simulate() -> tuple[np.ndarray, list[Phase]]:
    world = np.zeros(N_CELLS, dtype=bool)
    world[DOOR_CELLS] = True

    slip_p = (1.0 - P_MOVE_EXACT) / 2.0
    motion = hf.MotionModel(P_MOVE_EXACT, slip_p, slip_p)
    rng = np.random.default_rng(SEED)

    def read(pos: int) -> bool:
        truth = bool(world[pos])
        return truth if rng.random() < P_SENSOR_CORRECT else (not truth)

    pos = START_CELL
    belief = hf.uniform_belief(N_CELLS)
    phases: list[Phase] = [Phase("wake", 0, belief.copy(), belief.copy(), pos, pos)]

    # First observation, before moving at all.
    z = read(pos)
    after = hf.sense(belief, world, z, P_SENSOR_CORRECT, 1 - P_SENSOR_CORRECT)
    phases.append(Phase("sense", 0, belief.copy(), after.copy(), pos, pos,
                        reading=z, glitch=(z != bool(world[pos]))))
    belief = after

    for step in range(1, N_STEPS + 1):
        moved = motion.sample(COMMAND, rng)
        new_pos = (pos + moved) % N_CELLS
        after = motion.apply(belief, COMMAND)
        phases.append(Phase("move", step, belief.copy(), after.copy(), pos, new_pos,
                            slip=(moved != COMMAND)))
        belief, pos = after, new_pos

        z = read(pos)
        after = hf.sense(belief, world, z, P_SENSOR_CORRECT, 1 - P_SENSOR_CORRECT)
        phases.append(Phase("sense", step, belief.copy(), after.copy(), pos, pos,
                            reading=z, glitch=(z != bool(world[pos]))))
        belief = after

    return world, phases


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def build_figure(world: np.ndarray):
    fig = plt.figure(figsize=(11.0, 6.3), dpi=78, facecolor=SURFACE)

    ax_map = fig.add_axes([0.085, 0.545, 0.875, 0.265])
    ax_bel = fig.add_axes([0.085, 0.185, 0.875, 0.315])
    for ax in (ax_map, ax_bel):
        ax.set_facecolor(SURFACE)

    # ---- header -------------------------------------------------------
    fig.text(0.085, 0.945, "Histogram Filter  -  where am I in this corridor?",
             fontsize=15.5, fontweight="bold", color=INK, va="center")
    fig.text(0.96, 0.945, "Day 1 / Localization", fontsize=9.5, color=INK_3,
             va="center", ha="right")

    badge = fig.text(0.085, 0.868, " SENSE ", fontsize=9.5, fontweight="bold",
                     color=BELIEF, va="center", family="monospace",
                     bbox=dict(boxstyle="round,pad=0.42", fc=SENSE_BG, ec="none"))
    caption = fig.text(0.172, 0.868, "", fontsize=11, color=INK_2, va="center")

    # ---- corridor map -------------------------------------------------
    ax_map.set_xlim(-0.7, N_CELLS - 0.3)
    ax_map.set_ylim(-0.85, 2.55)
    ax_map.axis("off")

    for i in range(N_CELLS):
        is_door = bool(world[i])
        ax_map.add_patch(Rectangle((i - 0.45, 0), 0.9, 1.0,
                                   facecolor=DOOR if is_door else WALL,
                                   edgecolor=DOOR_DK if is_door else WALL_EDGE,
                                   linewidth=1.1, zorder=1))
        if is_door:                                  # a little door glyph
            ax_map.add_patch(Rectangle((i - 0.27, 0.13), 0.54, 0.74,
                                       facecolor=SURFACE, edgecolor=DOOR_DK,
                                       linewidth=0.9, zorder=2))
            ax_map.add_patch(Circle((i + 0.14, 0.5), 0.045, facecolor=DOOR_DK,
                                    edgecolor="none", zorder=3))

    # the corridor is cyclic - mass that slides off one end reappears at the other
    for x, sym in ((-0.62, "↩"), (N_CELLS - 0.38, "↪")):
        ax_map.text(x, 0.5, sym, fontsize=13, color=INK_3, ha="center", va="center")

    # robot + its sensor beam, both moved every frame
    beam = Line2D([], [], color=TRUTH, lw=1.4, ls=(0, (2.5, 2.0)), zorder=4, alpha=0.9)
    ax_map.add_line(beam)
    robot = Line2D([START_CELL], [1.62], marker="o", markersize=16, color=TRUTH,
                   markeredgecolor=SURFACE, markeredgewidth=2.2, zorder=6, ls="none")
    ax_map.add_line(robot)
    reading_txt = ax_map.text(START_CELL, 2.28, "", fontsize=9, fontweight="bold",
                              ha="center", va="center", color=INK, zorder=7,
                              bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE,
                                        ec=INK_3, lw=0.8))

    # Hand-placed legend in the band below the corridor, so the robot's speech
    # bubble can never land on top of it.
    for mx, mk, fc, ec, label in (
        (-0.35, "s", DOOR, DOOR_DK, "door - a landmark on the map"),
        (4.20, "s", WALL, WALL_EDGE, "blank wall"),
        (6.80, "o", TRUTH, SURFACE, "where the robot really is"),
    ):
        ax_map.plot([mx], [-0.48], marker=mk, markersize=8, markerfacecolor=fc,
                    markeredgecolor=ec, markeredgewidth=1.2, ls="none", clip_on=False)
        ax_map.text(mx + 0.32, -0.48, label, fontsize=8.5, color=INK_2,
                    va="center", ha="left")
    ax_map.text(N_CELLS - 0.35, -0.48,
                "the corridor wraps around: cell 19 is next to cell 0",
                fontsize=8, color=INK_3, ha="right", va="center", style="italic")

    # ---- belief bars --------------------------------------------------
    bars = ax_bel.bar(np.arange(N_CELLS), np.zeros(N_CELLS), width=0.8,
                      color=BELIEF, edgecolor="none", zorder=3)
    ax_bel.axhline(1.0 / N_CELLS, color=INK_3, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax_bel.text(N_CELLS - 0.55, 1.0 / N_CELLS + 0.085,
                "1/20 - the \"I have no idea\" line", fontsize=8, color=INK_3,
                ha="right", va="bottom", style="italic")

    ax_bel.set_xlim(-0.7, N_CELLS - 0.3)
    ax_bel.set_ylim(0, 1.0)
    ax_bel.set_xticks(range(N_CELLS))
    ax_bel.set_xticklabels(range(N_CELLS), fontsize=8.5, color=INK_2)
    ax_bel.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_bel.set_yticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=8.5, color=INK_2)
    ax_bel.set_xlabel("corridor cell", fontsize=9.5, color=INK_2, labelpad=4)
    ax_bel.set_ylabel("belief", fontsize=10, color=INK_2, labelpad=6)
    ax_bel.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax_bel.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax_bel.spines[side].set_visible(False)
    ax_bel.spines["bottom"].set_color(GRID)

    peak_txt = ax_bel.text(0, 0, "", fontsize=9.5, fontweight="bold", ha="center",
                           va="bottom", color=INK, zorder=5)
    # Figure-level so a tall bar can never grow into it.
    fig.legend(handles=[
        Line2D([], [], marker="s", ls="none", markersize=9, color=BELIEF,
               label="belief: how sure the robot is it is in this cell"),
        Line2D([], [], marker="s", ls="none", markersize=9, color=TRUTH,
               label="the cell the robot is actually in"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.118), frameon=False,
        fontsize=8.5, labelcolor=INK_2, handletextpad=0.4, ncol=2,
        columnspacing=2.2)

    footer = fig.text(0.085, 0.035, "", fontsize=9.5, color=INK_2, va="center",
                      family="monospace")

    return dict(fig=fig, ax_map=ax_map, ax_bel=ax_bel, bars=bars, robot=robot,
                beam=beam, reading_txt=reading_txt, badge=badge, caption=caption,
                peak_txt=peak_txt, footer=footer)


def render_frame(art, belief, pos_f, phase, show_reading):
    """Draw one frame. `pos_f` is a float so the robot can slide between cells."""
    cell = int(round(pos_f)) % N_CELLS

    for i, bar in enumerate(art["bars"]):
        bar.set_height(belief[i])
        bar.set_color(TRUTH if i == cell else BELIEF)

    art["robot"].set_data([pos_f], [1.62])

    if show_reading and phase.reading is not None:
        word = "I see a DOOR" if phase.reading else "I see a blank WALL"
        art["reading_txt"].set_text(word)
        art["reading_txt"].set_position((pos_f, 2.28))
        art["reading_txt"].get_bbox_patch().set_edgecolor(DOOR_DK if phase.reading else INK_3)
        art["reading_txt"].set_visible(True)
        art["beam"].set_data([pos_f, pos_f], [1.44, 1.04])
        art["beam"].set_visible(True)
    else:
        art["reading_txt"].set_visible(False)
        art["beam"].set_visible(False)

    peak = int(belief.argmax())
    if belief[peak] > 0.14:
        art["peak_txt"].set_text(f"{belief[peak] * 100:.0f}%")
        art["peak_txt"].set_position((peak, belief[peak] + 0.018))
        art["peak_txt"].set_color(TRUTH if peak == cell else BELIEF)
        art["peak_txt"].set_visible(True)
    else:
        art["peak_txt"].set_visible(False)

    art["footer"].set_text(
        f"best guess: cell {peak:<2d} ({belief[peak] * 100:4.1f}%)   "
        f"truth: cell {cell:<2d}   "
        f"uncertainty: {hf.entropy(belief):4.2f} bits  (started at {np.log2(N_CELLS):.2f})"
    )


def caption_for(phase: Phase) -> tuple[str, str, str, str]:
    """-> (badge text, badge colour, badge background, caption)"""
    if phase.kind == "wake":
        return (" START ", INK_2, "#efeeea",
                "The robot switches on somewhere in the corridor. Every cell is equally likely.")
    if phase.kind == "sense":
        seen = "a DOOR" if phase.reading else "a blank WALL"
        txt = (f"step {phase.step} - the sensor reports {seen}. "
               f"Boost cells that agree with the map, shrink the rest.")
        if phase.glitch:
            txt = (f"step {phase.step} - the sensor reports {seen}, but it is WRONG "
                   f"(cell {phase.pos_to} really is a door). Watch the filter cope.")
        return (" SENSE ", BELIEF, SENSE_BG, txt)
    txt = (f"step {phase.step} - drive 1 cell right. Slide the belief across "
           f"and blur it: the wheels might slip.")
    if phase.slip:
        txt = (f"step {phase.step} - drive 1 cell right ... but the wheels SLIPPED and "
               f"the robot never moved. It cannot tell; the blur covers it.")
    return (" MOVE  ", TRUTH, MOVE_BG, txt)


def make_gif(world, phases, out_path: Path, fps=14, ease=5, hold=3, tail=22):
    art = build_figure(world)
    fig = art["fig"]
    frames: list[Image.Image] = []

    def grab():
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))

    for idx, phase in enumerate(phases):
        btxt, bcol, bbg, cap = caption_for(phase)
        art["badge"].set_text(btxt)
        art["badge"].set_color(bcol)
        art["badge"].get_bbox_patch().set_facecolor(bbg)
        art["caption"].set_text(cap)

        n_ease = 1 if phase.kind == "wake" else ease
        n_hold = (hold + 6) if phase.kind == "wake" else hold
        if phase.glitch or phase.slip:
            n_hold += 5                      # linger on the interesting failures
        if idx == len(phases) - 1:
            n_hold += tail

        delta = phase.pos_to - phase.pos_from
        if delta > N_CELLS / 2:                 # took the short way round
            delta -= N_CELLS
        elif delta < -N_CELLS / 2:
            delta += N_CELLS

        for f in range(n_ease):
            t = smoothstep((f + 1) / n_ease)
            belief = (1 - t) * phase.belief_from + t * phase.belief_to
            pos_f = (phase.pos_from + t * delta) % N_CELLS
            # the reading is only "known" once a SENSE beat is under way
            render_frame(art, belief, pos_f, phase, show_reading=(phase.kind == "sense"))
            grab()
        for _ in range(n_hold):
            frames.append(frames[-1])

    plt.close(fig)

    # Share one adaptive palette across every frame: smaller file, no flicker.
    sample = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    for k, f in enumerate((frames[0], frames[len(frames) // 2], frames[-1])):
        sample.paste(f, (0, k * frames[0].height))
    palette = sample.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(out_path, save_all=True, append_images=quantized[1:],
                      duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    return out_path, len(frames)


def make_summary_png(world, phases, out_path: Path):
    """One still image of the whole run: every belief, stacked by step."""
    senses = [p for p in phases if p.kind == "sense"]
    grid = np.array([p.belief_to for p in senses])
    truth = [p.pos_to for p in senses]

    ramp = LinearSegmentedColormap.from_list(
        "belief", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])

    fig, ax = plt.subplots(figsize=(10.2, 5.0), dpi=110, facecolor=SURFACE)
    im = ax.imshow(grid, cmap=ramp, aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest", origin="upper")
    ax.plot(truth, range(len(truth)), marker="o", ms=7, color=TRUTH, lw=1.6,
            markeredgecolor=SURFACE, markeredgewidth=1.4,
            label="where the robot really was")

    for i in DOOR_CELLS:
        ax.add_patch(Rectangle((i - 0.5, -1.02), 1.0, 0.55, facecolor=DOOR,
                               edgecolor=DOOR_DK, lw=0.8, clip_on=False))
    ax.text(-1.3, -0.75, "map:", fontsize=9, color=INK_2, ha="right", va="center")

    ax.set_xticks(range(N_CELLS))
    ax.set_xticklabels(range(N_CELLS), fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(senses)))
    ax.set_yticklabels([f"step {p.step}" for p in senses], fontsize=8.5, color=INK_2)
    ax.set_xlabel("corridor cell", fontsize=10, color=INK_2)
    ax.set_title("The belief funnel - 20 maybes collapse into one answer",
                 fontsize=13, fontweight="bold", color=INK, pad=30, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.085), frameon=False,
              fontsize=9, labelcolor=INK_2)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
    cb.set_label("belief", fontsize=9, color=INK_2)
    cb.ax.tick_params(labelsize=8, length=0, colors=INK_2)
    cb.outline.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_trace(world, phases):
    print(f"map:  {''.join('D' if c else '.' for c in world)}   (D = door)")
    print(f"      {''.join(str(i % 10) for i in range(N_CELLS))}\n")
    for p in phases:
        if p.kind == "move":
            # Moving always *loses* information - show the entropy going back up.
            h0, h1 = hf.entropy(p.belief_from), hf.entropy(p.belief_to)
            slip = "   <- WHEEL SLIP (robot did not actually move)" if p.slip else ""
            print(f"         move  ->  blur    "
                  f"{' ' * N_CELLS}    H {h0:.2f} -> {h1:.2f}  ({h1 - h0:+.2f}){slip}")
        elif p.kind == "sense":
            b = p.belief_to
            bar = "".join("#" if v > .30 else "+" if v > .10 else "-" if v > .03 else "."
                          for v in b)
            glitch = "  <- SENSOR GLITCH" if p.glitch else ""
            print(f"step {p.step:2d}  true={p.pos_to:2d}  saw={'DOOR' if p.reading else 'wall'}  "
                  f"|{bar}|  p(truth)={b[p.pos_to]:.3f}  H={hf.entropy(b):.2f}{glitch}")


def main():
    ap = argparse.ArgumentParser(description="Day 1 histogram filter demo")
    ap.add_argument("--no-gif", action="store_true", help="skip the (slow) GIF render")
    args = ap.parse_args()

    world, phases = simulate()
    print_trace(world, phases)

    media = Path(__file__).parent / "media"
    png = make_summary_png(world, phases, media / "belief_evolution.png")
    print(f"\nwrote {png}")
    if not args.no_gif:
        gif, n = make_gif(world, phases, media / "histogram_filter_1d.gif")
        print(f"wrote {gif}  ({n} frames, {gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
