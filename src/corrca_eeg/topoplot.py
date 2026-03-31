from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray
from scipy.interpolate import griddata


FloatArray = NDArray[np.float64]


def read_loc_file(loc_path: str | Path) -> tuple[FloatArray, FloatArray, list[str]]:
    """Read an EEGLAB-style .loc file.

    Expected columns are channel index, angle in degrees, radius, and label.

    Args:
            loc_path: Path to a .loc file.

    Returns:
            Tuple of angles in degrees, radii, and labels.
    """
    loc_path = Path(loc_path)
    angles: list[float] = []
    radii: list[float] = []
    labels: list[str] = []

    with loc_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            angles.append(float(parts[1]))
            radii.append(float(parts[2]))
            labels.append(parts[3].rstrip("."))

    return np.asarray(angles, dtype=np.float64), np.asarray(radii, dtype=np.float64), labels


def _polar_to_cartesian(angles_deg: FloatArray, radii: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Convert topoplot polar coordinates to Cartesian.

    Angle 0 degrees points toward the nose (top), positive is toward right.

    Args:
            angles_deg: Channel angles in degrees.
            radii: Channel radii.

    Returns:
            Tuple of x and y coordinates.
    """
    theta = np.deg2rad(angles_deg)
    x = radii * np.sin(theta)
    y = radii * np.cos(theta)
    return x, y


def _draw_head(ax: Axes, headrad: float = 0.5, color: str = "k") -> None:
    """Draw the standard head circle, nose, and ears."""
    ax.add_patch(Circle((0.0, 0.0), headrad, fill=False, ec=color, lw=1.5))

    nose_x = np.array([-0.08, 0.0, 0.08])
    nose_y = np.array([headrad, headrad + 0.08, headrad])
    ax.plot(nose_x, nose_y, color=color, lw=1.5)

    left_ear_x = np.array([-headrad, -headrad - 0.03, -headrad - 0.03, -headrad])
    left_ear_y = np.array([0.08, 0.05, -0.05, -0.08])
    right_ear_x = -left_ear_x
    right_ear_y = left_ear_y
    ax.plot(left_ear_x, left_ear_y, color=color, lw=1.2)
    ax.plot(right_ear_x, right_ear_y, color=color, lw=1.2)


def topoplot(
    data_vector: FloatArray,
    chan_locs: str | Path,
    *,
    electrodes: str = "on",
    maplimits: str | tuple[float, float] = "absmax",
    style: str = "both",
    numcontour: int = 6,
    gridscale: int = 67,
    headrad: float = 0.5,
    cmap: str = "RdBu_r",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a scalp topography similar to EEGLAB topoplot.

    Args:
            data_vector: One value per channel.
            chan_locs: Path to an EEGLAB-style .loc file.
            electrodes: Electrode display mode: on, off, labels, or numbers.
            maplimits: Color scaling, one of absmax, maxmin, or explicit (min, max).
            style: Plot style: both, map, contour, or blank.
            numcontour: Number of contour lines.
            gridscale: Interpolation grid size.
            headrad: Head radius in topoplot coordinate system.
            cmap: Matplotlib colormap name.
            ax: Optional target axes.

    Returns:
            Figure and axes containing the topoplot.
    """
    values = np.asarray(data_vector, dtype=np.float64).ravel()
    angles_deg, radii, labels = read_loc_file(chan_locs)

    if values.size != angles_deg.size:
        raise ValueError(f"Expected {angles_deg.size} channel values based on loc file, got {values.size}.")

    x, y = _polar_to_cartesian(angles_deg, radii)

    xi = np.linspace(-headrad, headrad, gridscale)
    yi = np.linspace(-headrad, headrad, gridscale)
    xmesh, ymesh = np.meshgrid(xi, yi)

    zmesh = griddata((x, y), values, (xmesh, ymesh), method="cubic")
    if np.any(np.isnan(zmesh)):
        zmesh_linear = griddata((x, y), values, (xmesh, ymesh), method="linear")
        zmesh = np.where(np.isnan(zmesh), zmesh_linear, zmesh)
    if np.any(np.isnan(zmesh)):
        zmesh_nearest = griddata((x, y), values, (xmesh, ymesh), method="nearest")
        zmesh = np.where(np.isnan(zmesh), zmesh_nearest, zmesh)

    mask = (xmesh**2 + ymesh**2) <= (headrad**2)
    zmesh = np.where(mask, zmesh, np.nan)

    if maplimits == "absmax":
        vmax = float(np.nanmax(np.abs(values)))
        vmin = -vmax
    elif maplimits == "maxmin":
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    elif isinstance(maplimits, tuple) and len(maplimits) == 2:
        vmin, vmax = float(maplimits[0]), float(maplimits[1])
    else:
        raise ValueError("maplimits must be 'absmax', 'maxmin', or a (vmin, vmax) tuple.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    style = style.lower()
    if style in {"both", "map", "fill"}:
        ax.contourf(xmesh, ymesh, zmesh, levels=64, cmap=cmap, vmin=vmin, vmax=vmax)

    if style in {"both", "contour", "fill"}:
        ax.contour(xmesh, ymesh, zmesh, levels=numcontour, colors="0.25", linewidths=0.6)

    if style == "blank":
        pass

    electrodes_mode = electrodes.lower()
    if electrodes_mode == "on":
        ax.scatter(x, y, s=12, c="k", zorder=5)
    elif electrodes_mode == "labels":
        for xc, yc, label in zip(x, y, labels, strict=True):
            ax.text(xc, yc, label, ha="center", va="center", fontsize=7, zorder=6)
    elif electrodes_mode == "numbers":
        for idx, (xc, yc) in enumerate(zip(x, y, strict=True), start=1):
            ax.text(xc, yc, str(idx), ha="center", va="center", fontsize=7, zorder=6)
    elif electrodes_mode == "off":
        pass
    else:
        raise ValueError("electrodes must be one of: on, off, labels, numbers")

    _draw_head(ax=ax, headrad=headrad)

    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.62, 0.62)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    return fig, ax


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo_values = rng.normal(size=64)
    fig, _ = topoplot(demo_values, Path("data/corrca_data/BioSemi64.loc"), electrodes="off")
    fig.suptitle("Demo Topoplot")
    plt.show()
