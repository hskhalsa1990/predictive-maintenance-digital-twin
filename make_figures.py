"""Render static 3D figures from the project's own models (used in README.md).

Run:  python make_figures.py          (writes fig_*.png into the current folder)
Needs numpy, pandas, scipy and matplotlib. Everything is synthetic or model-derived
from educational linear models; the figures are illustrations, not measurements.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monitoring import generate_sensor_batch
from v10_enterprise import score_enterprise
from signal_diagnostics import generate_vibration_signal
from v17_rotor_lab import waterfall_spectra
from v19_advanced_rotor import build_timoshenko_rotor, solve_rotor_campbell
from v20_bearing_response import synchronous_unbalance_response

BG, FG, GRID = "#0f172a", "#e2e8f0", "#334155"


def _style(ax, title, xl, yl, zl):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=13, pad=12)
    for a, lab in ((ax.xaxis, xl), (ax.yaxis, yl), (ax.zaxis, zl)):
        a.set_label_text(lab, color=FG)
        a.set_pane_color((0.12, 0.16, 0.25, 1.0))
        a._axinfo["grid"]["color"] = GRID
    ax.tick_params(colors=FG, labelsize=8)


def _fig():
    fig = plt.figure(figsize=(8, 5.6), dpi=110, facecolor=BG)
    return fig, fig.add_subplot(111, projection="3d")


def _save(fig, name, out):
    fig.text(0.5, 0.015, "Synthetic / educational model output. Not a measurement or certified analysis.",
             ha="center", color="#94a3b8", fontsize=8)
    fig.savefig(out / name, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def campbell_3d(out):
    rotor = build_timoshenko_rotor()
    camp = solve_rotor_campbell(rotor, np.linspace(0, 12000, 49))
    camp = camp[camp.frequency_hz < 900]
    fig, ax = _fig()
    for whirl, color in (("Forward", "#38bdf8"), ("Backward", "#f87171")):
        g = camp[camp.whirl == whirl]
        ax.scatter(g.rpm, g.frequency_hz, g.damping_ratio.clip(-0.05, 0.5), s=14, c=color, label=f"{whirl} whirl", depthshade=True)
    rpm = np.linspace(0, 12000, 2)
    ax.plot(rpm, rpm / 60.0, np.zeros(2), color="#fbbf24", lw=1.8, label="1X synchronous line")
    _style(ax, "Gyroscopic Campbell diagram in 3D (Timoshenko rotor)", "Speed (RPM)", "Frequency (Hz)", "Damping ratio")
    ax.view_init(24, -58); leg = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8, loc="upper left")
    _save(fig, "fig_campbell_3d.png", out)


def unbalance_surface_3d(out):
    rpm = np.linspace(300, 9000, 60); ks = np.geomspace(5e6, 1e8, 26)
    z = np.empty((len(ks), len(rpm)))
    for i, k in enumerate(ks):
        resp = synchronous_unbalance_response(build_timoshenko_rotor(bearing_stiffness_n_m=float(k)), rpm, 1e-4)
        z[i] = resp.resultant_amplitude_m.to_numpy() * 1e6
    X, Y = np.meshgrid(rpm, np.log10(ks))
    fig, ax = _fig()
    surf = ax.plot_surface(X, Y, np.log10(z), cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    _style(ax, "1X unbalance response vs bearing stiffness", "Speed (RPM)", "log10 bearing stiffness (N/m)", "log10 amplitude (µm)")
    ax.view_init(26, -60)
    cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08); cb.ax.tick_params(colors=FG, labelsize=8)
    _save(fig, "fig_unbalance_surface_3d.png", out)


def waterfall_3d(out):
    fs = 2048; rpms = np.linspace(600, 3600, 16)
    sigs = [generate_vibration_signal(2.0, fs, r / 60.0, 180.0, 0.05, 7 + i)[1] for i, r in enumerate(rpms)]
    f, r, psd = waterfall_spectra(sigs, fs, rpms, nperseg=1024)
    m = f <= 400; amp = np.sqrt(psd[:, m])
    X, Y = np.meshgrid(f[m], r)
    fig, ax = _fig()
    surf = ax.plot_surface(X, Y, amp, cmap="plasma", linewidth=0, antialiased=True)
    _style(ax, "Vibration waterfall: spectrum vs shaft speed", "Frequency (Hz)", "Speed (RPM)", "Amplitude (a.u.)")
    ax.view_init(30, -62)
    cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08); cb.ax.tick_params(colors=FG, labelsize=8)
    _save(fig, "fig_waterfall_3d.png", out)


def fleet_3d(out):
    frames = [score_enterprise(generate_sensor_batch(b, machines=8, seed=2026 + b), 2026 + b) for b in range(18)]
    d = pd.concat(frames, ignore_index=True)
    colors = d.risk_band.map({"Low": "#22c55e", "Guarded": "#facc15", "High": "#fb923c", "Critical": "#ef4444"}).fillna("#94a3b8")
    fig, ax = _fig()
    ax.scatter(d.temperature_c, d.vibration_rms, d.health_index, c=colors, s=26, depthshade=True, edgecolors="none")
    _style(ax, "Fleet health digital twin (synthetic batches)", "Temperature (°C)", "Vibration RMS", "Health index")
    ax.view_init(22, -55)
    for label, c in (("Low", "#22c55e"), ("Guarded", "#facc15"), ("High", "#fb923c"), ("Critical", "#ef4444")):
        ax.scatter([], [], [], c=c, label=f"{label} risk band")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8, loc="upper left")
    _save(fig, "fig_fleet_3d.png", out)


def main(argv=None) -> int:
    out = Path(argv[0]) if argv else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    for fn in (campbell_3d, unbalance_surface_3d, waterfall_3d, fleet_3d):
        fn(out); print("wrote", fn.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
