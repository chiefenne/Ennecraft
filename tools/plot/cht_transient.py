"""Transient conjugate heat transfer of stage 7d6 (gallery entry of the PR).

cht_heated_wall_transient.png: the heated wall of 7d1 with the heater
switched on at t = 0 (check V7.37), the temperature at three times (left)
and the heat balance of every step (right).
cht_contact_transient.png: two bodies brought into contact (check V7.6),
the temperature along the bar at three times against the exact erf
profiles (left) and the error over the mesh size (right).

    pip install matplotlib h5py numpy
    python3 -m tools.plot.cht_transient OUT_DIR WALL_RUN CONTACT_RUN SPACE_RUN TIME_RUN

with the output directories of

    target/release/foehn flow examples/cases/flow/cht_heated_wall_transient.toml --out output/cht_wall_t
    target/release/foehn flow examples/cases/flow/cht_contact_transient.toml --out output/cht_contact
    target/release/foehn flow examples/cases/flow/cht_contact_transient.toml --out output/cht_contact_space \\
        --set 'verification.levels=[0, 1, 2]' --set 'mesh.generate.cells=[100, 1, 1]'
    target/release/foehn flow examples/cases/flow/cht_contact_transient.toml --out output/cht_contact_time \\
        --set 'verification.time_steps=[20.0, 10.0, 5.0, 2.5]'
"""
import json
import math
import os
import sys
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from tools.plot.energy_fields import line
from tools.plot.turbulence_fields import read


def time_step(run):
    with open(os.path.join(run, "case.resolved.toml"), "rb") as f:
        return tomllib.load(f)["time"]["time_step"]


def fields_at(run, step):
    return read(os.path.join(run, f"fields_{step:06d}.vtkhdf"))


def wall(out, run):
    dt = time_step(run)
    steps = (60, 180, 600)
    fig = plt.figure(figsize=(15.0, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1])
    pc = None
    for i, n in enumerate(steps):
        quads, _, d = fields_at(run, n)
        ax = fig.add_subplot(gs[i, 0])
        pc = PolyCollection(
            quads, array=d["temperature"], cmap="inferno", edgecolors="face", linewidths=0.2
        )
        pc.set_clim(300.0, 324.0)
        ax.add_collection(pc)
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 1.2)
        ax.set_aspect("equal")
        ax.axhline(0.2, color="c", lw=0.9, ls="--")
        ax.text(0.05, 1.1, f"t = {n * dt:.0f} s", color="w", va="top", fontsize=10)
        ax.set_ylabel("y (m)")
        if i == 0:
            ax.set_title("Heater switched on at t = 0: 1000 W/m² into the wall from below")
        if i < len(steps) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("x (m)")
    fig.colorbar(pc, ax=fig.axes, label="temperature (K)", shrink=0.9, pad=0.01, location="left")
    h = line(os.path.join(run, "heat.csv"))
    ax = fig.add_subplot(gs[:, 1])
    t = h["time"]
    ax.plot(t, h["wall_storage"], "-", color="#d95f02", lw=1.8, label="stored in the wall")
    ax.plot(t, h["interface-wall_heat"], "-", color="#1b9e77", lw=1.8,
            label="through the interface into the fluid")
    ax.plot(t, -h["convection"], "-", color="#7570b3", lw=1.8, label="carried out by the flow")
    ax.axhline(160.0, color="0.4", lw=0.8, ls=":")
    ax.text(0.98 * t[-1], 162.0, "heater 160 W", color="0.3", ha="right", va="bottom", fontsize=9)
    ax.set_xlim(0.0, t[-1])
    ax.set_ylim(0.0, 175.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("heat flow (W)")
    worst = h["relative"].max()
    ax.set_title(f"Heat balance of every step: largest imbalance {worst:.1e} of the heat flow")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out, "cht_heated_wall_transient.png"), dpi=105)


def exact(x, t):
    k1, k2, c = 10.0, 1.0, 1e6
    e1, e2 = math.sqrt(k1 * c), math.sqrt(k2 * c)
    ti = (e1 * 400.0 + e2 * 300.0) / (e1 + e2)
    return np.where(
        x < 0.5,
        ti + (400.0 - ti) * np.vectorize(math.erf)((0.5 - x) / (2 * np.sqrt(k1 / c * t))),
        ti + (300.0 - ti) * np.vectorize(math.erf)((x - 0.5) / (2 * np.sqrt(k2 / c * t))),
    )


def contact(out, run, space, time):
    dt = time_step(run)
    fig, axes = plt.subplots(
        1, 2, figsize=(14.0, 4.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]}
    )
    ax = axes[0]
    xs = np.linspace(0.3, 0.7, 801)
    for n, c in ((50, "#7570b3"), (150, "#1b9e77"), (500, "#d95f02")):
        quads, _, d = fields_at(run, n)
        x = quads.mean(axis=1)[:, 0]
        o = np.argsort(x)
        x, tc = x[o], d["temperature"][o]
        sel = (x > 0.3) & (x < 0.7)
        ax.plot(x[sel][::4], tc[sel][::4], "o", color=c, ms=3.5, label=f"t = {n * dt:.0f} s")
        ax.plot(xs, exact(xs, n * dt), "-", color=c, lw=1.0)
    ax.axvline(0.5, color="0.4", lw=0.8, ls="--")
    ax.text(0.49, 302, "solid, k = 10 W/(m K)", ha="right", fontsize=9)
    ax.text(0.51, 392, "fluid at rest, k = 1 W/(m K)", ha="left", fontsize=9)
    ax.set_xlim(0.3, 0.7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("temperature (K)")
    ax.set_title("Two bodies at 400 K and 300 K brought into contact: cells (dots), exact erf (lines)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax = axes[1]
    with open(os.path.join(space, "flow.json")) as f:
        rep = json.load(f)
    pts = []
    for s in rep["solves"]:
        t = next(x for x in s["scalars"] if x["name"] == "temperature")
        pts.append((t["h"], t["l2"]))
    h, e = np.array(pts).T
    ax.loglog(h, e, "o-", color="#d95f02", lw=1.6, label="L2 error against the exact solution")
    ax.loglog(h, 0.6 * e[-1] * (h / h[-1]) ** 2, ":", color="0.3", lw=1.2, label="slope 2")
    so = dict((n, o) for n, o in rep["scalar_orders"])["temperature"]
    with open(os.path.join(time, "flow.json")) as f:
        to = dict((n, o) for n, o in json.load(f)["scalar_orders"])["temperature (time)"]
    ax.text(
        0.04, 0.96,
        f"observed order in space {so[-1]:.2f}\nin time (BDF2, Δt = 20 to 2.5 s) {to[-1]:.2f}",
        transform=ax.transAxes, va="top", fontsize=9,
    )
    ax.set_xlabel("cell size h (m)")
    ax.set_ylabel("L2 error of the temperature at t = 500 s (K)")
    ax.set_title("Second order in space and time")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(out, "cht_contact_transient.png"), dpi=110)


def main():
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    out, wall_run, run, space, time = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    wall(out, wall_run)
    contact(out, run, space, time)


if __name__ == "__main__":
    main()
