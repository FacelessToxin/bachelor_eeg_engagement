from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.linalg import eig
from scipy.signal import butter, lfilter, sosfilt


FloatArray = NDArray[np.float64]


def _to_python_indices(indices: NDArray[np.float64] | list[int] | None, n_channels: int) -> NDArray[np.int64]:
    """Convert channel indices to 0-based integer indices.

    Args:
            indices: Input indices, possibly from MATLAB (1-based).
            n_channels: Number of available channels.

    Returns:
            Zero-based integer indices constrained to valid channel range.
    """
    if indices is None:
        return np.array([], dtype=np.int64)

    arr = np.asarray(indices).astype(np.int64).ravel()
    if arr.size == 0:
        return np.array([], dtype=np.int64)

    if np.any(arr == 0):
        idx = arr
    else:
        idx = arr - 1

    idx = idx[(idx >= 0) & (idx < n_channels)]
    return np.unique(idx)


def _compute_rij(x: FloatArray) -> FloatArray:
    """Compute subject-by-subject cross-covariance tensor.

    Args:
            x: EEG data with shape (T, D, N).

    Returns:
            Covariance tensor Rij with shape (D, D, N, N).
    """
    t, d, n = x.shape
    x_2d = np.reshape(x, (t, d * n), order="F")
    cov_mat = np.cov(x_2d, rowvar=False)
    rij = np.reshape(cov_mat, (d, n, d, n), order="F")
    return np.transpose(rij, (0, 2, 1, 3))


def preprocess(
    x: FloatArray,
    eogchannels: NDArray[np.int64],
    badchannels: list[int | NDArray[np.int64]],
    fs: float,
) -> FloatArray:
    """Apply EEG preprocessing that matches the original MATLAB pipeline.

    Args:
            x: EEG data with shape (T, D, N).
            eogchannels: 0-based EOG channel indices.
            badchannels: Per-subject bad-channel specification; `-1` means auto-detect.
            fs: Sampling rate in Hz.

    Returns:
            Preprocessed EEG with EOG channels removed.
    """
    k_iqd = 4.0
    k_iqdp = 3.0
    hp_cutoff = 0.5

    _, d, n = x.shape
    sos = butter(5, hp_cutoff / (fs / 2.0), btype="high", output="sos")

    eegchannels = np.setdiff1d(np.arange(d, dtype=np.int64), eogchannels)

    for i in range(n):
        data = x[:, :, i].copy()
        t = data.shape[0]

        data = data - np.tile(data[0:1, :], (t, 1))
        data = sosfilt(sos, data, axis=0)

        if eogchannels.size > 0:
            x_eog = data[:, eogchannels]
            beta, *_ = np.linalg.lstsq(x_eog, data, rcond=None)
            data = data - x_eog @ beta

        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        iqd = q75 - q25
        thresh = k_iqd * np.tile(iqd.reshape(1, -1), (t, 1))
        data[np.abs(data) > thresh] = np.nan

        h_len = max(1, int(round(0.04 * fs)))
        h = np.zeros(h_len, dtype=np.float64)
        h[0] = 1.0
        data = np.flipud(lfilter(h, 1.0, np.flipud(lfilter(h, 1.0, data, axis=0)), axis=0))

        data[np.isnan(data)] = 0.0

        if isinstance(badchannels[i], int) and badchannels[i] == -1:
            logpower = np.log(np.std(data, axis=0) + 1e-12)
            eeg_logpower = np.log(np.std(data[:, eegchannels], axis=0) + 1e-12)
            q = np.percentile(eeg_logpower, [25, 50, 75])
            outlier_mask = (logpower - q[1]) > (k_iqdp * (q[2] - q[0]))
            badchannels[i] = np.where(outlier_mask)[0].astype(np.int64)

        bad_i = np.asarray(badchannels[i], dtype=np.int64).ravel()
        bad_i = bad_i[(bad_i >= 0) & (bad_i < d)]
        if bad_i.size > 0:
            data[:, bad_i] = 0.0

        x[:, :, i] = data

    return x[:, eegchannels, :]


def phaserandomized(x: FloatArray) -> FloatArray:
    """Generate phase-randomized surrogate data.

    Args:
            x: EEG data with shape (T, D, N).

    Returns:
            Surrogate EEG data preserving amplitude spectrum and covariance structure.
    """
    t, d, n = x.shape
    tr = int(round(t / 2.0) * 2)
    xr = np.zeros_like(x)

    for i in range(n):
        xfft = np.fft.fft(x[:, :, i], n=tr, axis=0)
        amp = np.abs(xfft[: tr // 2 + 1, :])
        phi = np.angle(xfft[: tr // 2 + 1, :])
        phir = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, size=(tr // 2 - 1, 1))

        tmp_half = np.zeros((tr // 2 + 1, d), dtype=np.complex128)
        tmp_half[1 : tr // 2, :] = amp[1 : tr // 2, :] * np.exp(1j * (phi[1 : tr // 2, :] + phir))

        full_spec = np.vstack(
            [
                xfft[0:1, :],
                tmp_half[1 : tr // 2, :],
                xfft[tr // 2 : tr // 2 + 1, :],
                np.conj(tmp_half[tr // 2 - 1 : 0 : -1, :]),
            ]
        )
        tmp = np.fft.ifft(full_spec, axis=0)
        xr[:, :, i] = np.real(tmp[:t, :])

    return xr


def isceeg(datafile: str | Path | None = None) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Compute ISC and correlated components from EEG.

    Args:
            datafile: Path to MATLAB file containing `X`, `fs`, `eogchannels`, and `badchannels`.

    Returns:
            Tuple of ISC eigenvalues, ISC per subject, ISC per second, projection vectors W, and scalp projections A.
    """
    gamma = 0.1
    nsec = 5

    if datafile is None:
        candidates = [Path("EEGVolume.mat"), Path("data/corrca_data/EEGVolume.mat")]
        selected = next((p for p in candidates if p.exists()), candidates[0])
        datafile = selected
        print("Using demo data from Cohen et al. 2016")

    datafile = Path(datafile)

    if datafile.exists():
        mat = loadmat(datafile, squeeze_me=True)
        x = np.asarray(mat["X"], dtype=np.float64)
        fs = float(np.asarray(mat["fs"]).squeeze())

        if x.ndim != 3:
            raise ValueError("Expected X to have shape (T, D, N).")

        n_subjects = x.shape[2]
        eogchannels = _to_python_indices(mat.get("eogchannels", np.array([])), x.shape[1])
        badchannels: list[int | NDArray[np.int64]] = [-1 for _ in range(n_subjects)]
    else:
        print("Can not find data file. Using random data.")
        fs = 256.0
        x = np.random.randn(int(30 * fs), 64, 15).astype(np.float64)
        eogchannels = np.array([], dtype=np.int64)
        badchannels = [-1 for _ in range(x.shape[2])]

    x = preprocess(x, eogchannels, badchannels, fs)
    t, d, n = x.shape

    rij = _compute_rij(x)
    rw = np.mean(np.stack([rij[:, :, i, i] for i in range(n)], axis=2), axis=2)
    rb = (np.sum(rij, axis=(2, 3)) - (n * rw)) / ((n - 1) * n)

    rw_reg = ((1.0 - gamma) * rw) + (gamma * np.mean(np.linalg.eigvals(rw).real) * np.eye(d))

    isc_eigs, w = eig(rb, rw_reg)
    isc_vals = np.real_if_close(isc_eigs)
    order = np.argsort(isc_vals)[::-1]
    isc = np.real_if_close(isc_vals[order]).astype(np.float64)
    w = np.real_if_close(w[:, order]).astype(np.float64)

    a = rw @ w @ np.linalg.inv(w.T @ rw @ w)

    isc_persubject = np.zeros((d, n), dtype=np.float64)
    for i in range(n):
        rw_i = np.zeros((d, d), dtype=np.float64)
        for j in range(n):
            if i != j:
                rw_i = rw_i + (rij[:, :, i, i] + rij[:, :, j, j]) / (n - 1)

        rb_i = np.zeros((d, d), dtype=np.float64)
        for j in range(n):
            if i != j:
                rb_i = rb_i + (rij[:, :, i, j] + rij[:, :, j, i]) / (n - 1)

        num = np.diag(w.T @ rb_i @ w)
        den = np.diag(w.T @ rw_i @ w)
        isc_persubject[:, i] = np.real_if_close(num / den)

    n_windows = int(np.floor((t - nsec * fs) / fs))
    isc_persecond = np.zeros((d, max(n_windows, 0)), dtype=np.float64)

    for ti in range(n_windows):
        start = ti * int(fs)
        stop = start + (nsec * int(fs))
        xt = x[start:stop, :, :]

        rij_t = _compute_rij(xt)
        rw_t = np.mean(np.stack([rij_t[:, :, i, i] for i in range(n)], axis=2), axis=2)
        rb_t = (np.sum(rij_t, axis=(2, 3)) - (n * rw_t)) / ((n - 1) * n)

        num_t = np.diag(w.T @ rb_t @ w)
        den_t = np.diag(w.T @ rw_t @ w)
        isc_persecond[:, ti] = np.real_if_close(num_t / den_t)

    _ = phaserandomized(x)

    return isc, isc_persubject, isc_persecond, w, a


def _plot_notboxplot_subjects(
    ax,
    y: FloatArray,
    jitter: float = 0.3,
    style: str = "patch",
) -> None:
    """Plot a MATLAB notBoxPlot-style summary for ISC per-subject values.

    Args:
            ax: Matplotlib axis where data will be plotted.
            y: Data matrix with shape (samples, groups).
            jitter: Horizontal jitter amplitude for raw points.
            style: One of patch, line, sdline, mean, or median.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    if y.ndim != 2:
        raise ValueError("y must be 2D with shape (samples, groups)")

    style = style.lower()
    valid_styles = {"patch", "line", "sdline", "mean", "median"}
    if style not in valid_styles:
        raise ValueError(f"Invalid style '{style}'. Expected one of: {sorted(valid_styles)}")

    _, n_groups = y.shape
    x_positions = np.arange(1, n_groups + 1, dtype=np.float64)
    jit_scale = jitter * 0.55
    cols = plt.cm.hsv(np.linspace(0, 1, n_groups + 1))[:, :3] * 0.5
    cols[0, :] = 0.0

    for k, xk in enumerate(x_positions):
        vals = y[:, k]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue

        mu = float(np.nanmean(vals))
        if style == "median":
            mu = float(np.nanmedian(vals))
        # MATLAB std(...,0,...) uses sample standard deviation (N-1 normalization).
        sd = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
        sem95 = float(1.96 * sd / np.sqrt(vals.size))

        sd_color = np.array([0.6, 0.6, 1.0], dtype=np.float64)
        sem_color = np.array([1.0, 0.6, 0.6], dtype=np.float64)

        def _add_interval_patch(interval: float, color: FloatArray) -> None:
            lower = mu - interval
            upper = mu + interval
            poly = Polygon(
                [[xk - jit_scale, lower], [xk + jit_scale, lower], [xk + jit_scale, upper], [xk - jit_scale, upper]],
                closed=True,
                facecolor=color,
                edgecolor=color * 0.8,
                linewidth=1.0,
                alpha=0.9,
            )
            ax.add_patch(poly)

        if style == "patch":
            _add_interval_patch(sd, sd_color)
            _add_interval_patch(sem95, sem_color)
            ax.plot([xk - jit_scale, xk + jit_scale], [mu, mu], "-r", linewidth=2)
        elif style == "sdline":
            _add_interval_patch(sem95, sem_color)
            ax.plot([xk - jit_scale, xk + jit_scale], [mu, mu], "-r", linewidth=2)
            ax.plot([xk, xk], [mu - sd, mu + sd], color=[0.2, 0.2, 1.0], linewidth=2)
        elif style == "line":
            ax.plot([xk, xk], [mu - sd, mu + sd], color=[0.2, 0.2, 1.0], linewidth=2)
            ax.plot([xk, xk], [mu - sem95, mu + sem95], "-r", linewidth=2)
            ax.plot(xk, mu, "or", markersize=8, markerfacecolor="r")
        else:
            ax.plot([xk - jit_scale, xk + jit_scale], [mu, mu], "-k", linewidth=2)

        x_jitter = xk + (np.random.rand(vals.size) - 0.5) * jitter
        # Match MATLAB indexing where first group uses cols(1,:) = black.
        c = cols[min(k, cols.shape[0] - 1)]
        ax.plot(
            x_jitter,
            vals,
            "o",
            color=c,
            markerfacecolor=c + (1.0 - c) * 0.65,
            markersize=4,
        )

    ax.set_xlim(0, n_groups + 1)
    ax.set_xticks(x_positions)
    ax.set_xlabel("Component")
    ax.set_ylabel("ISC")
    ax.set_title("Per subject")
    ax.grid(axis="y", alpha=0.2)


if __name__ == "__main__":
    import argparse
    import os
    from datetime import datetime

    import matplotlib

    from corrca_eeg.topoplot import topoplot

    parser = argparse.ArgumentParser(description="Compute ISC and optionally display/save summary plots.")
    parser.add_argument(
        "--save-figure",
        action="store_true",
        help="Save the summary figure to reports/figures with a timestamped filename.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output filename for --save-figure (e.g., my_plot or my_plot.png).",
    )
    parser.add_argument(
        "--subject-plot",
        type=str,
        default="boxplot",
        choices=["boxplot", "notboxplot"],
        help="Choose per-subject ISC panel style.",
    )
    parser.add_argument(
        "--notbox-style",
        type=str,
        default="patch",
        choices=["patch", "line", "sdline", "mean", "median"],
        help="Style used when --subject-plot notboxplot.",
    )
    parser.add_argument(
        "--notbox-jitter",
        type=float,
        default=0.3,
        help="Jitter magnitude for notboxplot raw points (MATLAB default is 0.3).",
    )
    args = parser.parse_args()

    # Prefer an interactive backend for the default run mode (show figure).
    if not args.save_figure and "MPLBACKEND" not in os.environ:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass

    import matplotlib.pyplot as plt

    isc_vals, isc_subj, isc_sec, w_mat, a_mat = isceeg()
    print(
        f"ISC shape={isc_vals.shape}, ISC_persubject shape={isc_subj.shape}, "
        f"ISC_persecond shape={isc_sec.shape}, W shape={w_mat.shape}, A shape={a_mat.shape}"
    )

    ncomp = min(3, a_mat.shape[1], isc_subj.shape[0], isc_sec.shape[0])
    loc_candidates = [Path("BioSemi64.loc"), Path("data/corrca_data/BioSemi64.loc")]
    loc_path = next((p for p in loc_candidates if p.exists()), None)

    if loc_path is None:
        print("Could not find BioSemi64.loc, skipping topoplot display.")
    else:
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1])

        for i in range(ncomp):
            ax_topo = fig.add_subplot(gs[0, i])
            topoplot(a_mat[:, i], loc_path, electrodes="off", ax=ax_topo)
            ax_topo.set_title(f"a_{i + 1}")

        ax_subj = fig.add_subplot(gs[1, 0])
        if args.subject_plot == "boxplot":
            subj_data = [isc_subj[i, :] for i in range(ncomp)]
            ax_subj.boxplot(subj_data, showfliers=True)
            for i in range(ncomp):
                x_jitter = (i + 1) + 0.06 * np.random.randn(isc_subj.shape[1])
                ax_subj.plot(x_jitter, isc_subj[i, :], "o", markersize=3, alpha=0.45, color="0.25")
            ax_subj.set_xlabel("Component")
            ax_subj.set_ylabel("ISC")
            ax_subj.set_title("Per subject")
        else:
            subj_y = isc_subj[:ncomp, :].T
            _plot_notboxplot_subjects(ax_subj, subj_y, jitter=args.notbox_jitter, style=args.notbox_style)

        ax_time = fig.add_subplot(gs[1, 1:3])
        ax_time.plot(isc_sec[:ncomp, :].T)
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("ISC")
        ax_time.set_title("Per second")
        ax_time.legend([f"C{i + 1}" for i in range(ncomp)], loc="best")

        fig.tight_layout()
        if args.save_figure:
            if args.output_name is None:
                timestamp = datetime.now().strftime("%B_%d_%H_%M")
                output_name = f"isceeg_results_{timestamp}.png"
            else:
                output_name = args.output_name
                if not output_name.lower().endswith(".png"):
                    output_name = f"{output_name}.png"

            output_path = Path("reports/figures") / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {output_path}")
        else:
            if "agg" in plt.get_backend().lower():
                print("Non-interactive backend detected; figure cannot be shown in this session.")
                print("Use --save-figure to save output, or run with an interactive backend.")
            else:
                plt.show()
