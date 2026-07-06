#!/usr/bin/env python3
"""Compare crystalGUI Outputs PSD data against supervisor Excel reference.

Workflow
--------
1. Run batch inference on the Outputs tab for your dataset.
2. Click **Download Full JSON** to export summary + per-image stats.
3. Run this script, e.g.::

     python scripts/compare_psd.py \\
       --gui data/exports/outputs_reference_20x.json \\
       --excel "data/real_datasets/Insulin PSD for Caio.xlsx" \\
       --sheet "Reference-2.5mgmL, 20C, 400RPM" \\
       --um-per-px 0.16 \\
       --out data/psd_comparison/reference_20x

Use ``--fit-scale`` to estimate um-per-px by matching mean crystal length to the
supervisor time series (helpful when the pixel calibration is unknown).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SupervisorSeries:
    label: str
    times: np.ndarray
    mean_um: np.ndarray
    count: np.ndarray
    bin_edges_um: np.ndarray
    hist_by_time: List[np.ndarray]


@dataclass
class GuiSeries:
    label: str
    times: np.ndarray
    mean_length_px: np.ndarray
    mean_width_px: np.ndarray
    mean_ar: np.ndarray
    count_avg: np.ndarray
    lengths_by_time: Dict[float, np.ndarray]
    widths_by_time: Dict[float, np.ndarray]
    um_per_px_from_meta: Optional[float] = None


@dataclass
class SupervisorComparison:
    length_weighted: SupervisorSeries
    square_weighted: SupervisorSeries


def _weight_label_for_header(df: pd.DataFrame, header_row: int, time_col: int) -> str:
    for r in range(header_row - 1, max(-1, header_row - 6), -1):
        for c in range(max(0, time_col - 1), min(df.shape[1], time_col + 3)):
            v = str(df.iloc[r, c]).strip()
            if "weighted" in v.lower():
                return v
    return "Supervisor"


def _find_time_header_rows(df: pd.DataFrame) -> List[Tuple[int, int]]:
    hits = []
    for r in range(len(df)):
        for c in range(min(30, df.shape[1])):
            if str(df.iloc[r, c]).strip() == "Time (min)":
                hits.append((r, c))
    return hits


def _supervisor_plot_label(weight_label: str) -> str:
    wl = weight_label.strip().lower()
    if "length" in wl:
        return "Supervisor (length-weighted)"
    if "square" in wl or "area" in wl:
        return "Supervisor (square-weighted)"
    return f"Supervisor ({weight_label})"


def _parse_supervisor_table(df: pd.DataFrame, header_row: int, time_col: int, label: str) -> SupervisorSeries:
    bins_row = header_row - 2
    bin_edges: List[float] = []
    for c in range(time_col + 6, df.shape[1]):
        v = df.iloc[bins_row, c]
        if pd.isna(v):
            break
        try:
            bin_edges.append(float(v))
        except (TypeError, ValueError):
            break
    if len(bin_edges) < 2:
        raise ValueError(f"Could not parse bin edges for table '{label}'")

    mean_col = time_col + 4
    count_col = time_col + 5
    hist_start = time_col + 6
    n_bins = len(bin_edges) - 1

    times, means, counts, hists = [], [], [], []
    for r in range(header_row + 1, len(df)):
        t_raw = df.iloc[r, time_col]
        if pd.isna(t_raw):
            continue
        try:
            t = float(t_raw)
        except (TypeError, ValueError):
            break

        mean_v = df.iloc[r, mean_col]
        cnt_v = df.iloc[r, count_col]
        hist = []
        for c in range(hist_start, hist_start + n_bins):
            v = df.iloc[r, c]
            hist.append(0.0 if pd.isna(v) else float(v))

        if len(hist) < n_bins:
            hist.extend([0.0] * (n_bins - len(hist)))
        hist = hist[:n_bins]

        times.append(t)
        means.append(np.nan if pd.isna(mean_v) else float(mean_v))
        counts.append(np.nan if pd.isna(cnt_v) else float(cnt_v))
        hists.append(np.array(hist, dtype=float))

    if not times:
        raise ValueError(f"No time rows parsed for table '{label}'")

    return SupervisorSeries(
        label=label,
        times=np.asarray(times, dtype=float),
        mean_um=np.asarray(means, dtype=float),
        count=np.asarray(counts, dtype=float),
        bin_edges_um=np.asarray(bin_edges, dtype=float),
        hist_by_time=hists,
    )


def load_supervisor_excel(path: Path, sheet: str, table: str = "auto") -> SupervisorComparison:
    """Load length-weighted and square-weighted supervisor time series from a sheet."""
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    headers = _find_time_header_rows(df)
    if len(headers) < 2:
        raise ValueError(f"Expected length- and square-weighted tables in sheet '{sheet}'")

    if table == "microscopy" and len(headers) >= 4:
        # Optional lower section with paired weighting columns (e.g. row ~69).
        header_row, time_col = headers[2]
        square_header, square_col = headers[3]
        length_label = _weight_label_for_header(df, header_row, time_col)
        square_label = _weight_label_for_header(df, square_header, square_col)
    else:
        # Primary Blaze time-series blocks (rows ~5 and ~27 in reference sheets).
        header_row, time_col = headers[0]
        square_header, square_col = headers[1]
        length_label = _weight_label_for_header(df, header_row, time_col)
        square_label = _weight_label_for_header(df, square_header, square_col)

    length_series = _parse_supervisor_table(
        df, header_row, time_col, _supervisor_plot_label(length_label)
    )
    square_series = _parse_supervisor_table(
        df, square_header, square_col, _supervisor_plot_label(square_label)
    )
    return SupervisorComparison(length_weighted=length_series, square_weighted=square_series)


def load_gui_export(path: Path) -> GuiSeries:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary") or {}
    per_image = data.get("per_image") or []
    times = [float(t) for t in (summary.get("times") or [])]
    stats_by_time = summary.get("stats_by_time") or {}

    mean_len, mean_wid, mean_ar, count_avg = [], [], [], []
    lengths_by_time: Dict[float, List[float]] = {}
    widths_by_time: Dict[float, List[float]] = {}

    for t in times:
        key = str(int(t)) if float(t).is_integer() else str(t)
        if key not in stats_by_time:
            key = str(t)
        st = stats_by_time.get(key) or stats_by_time.get(f"{t:.0f}") or {}
        mean_len.append(float(st.get("mean_length", np.nan)))
        mean_wid.append(float(st.get("mean_width", np.nan)))
        mean_ar.append(float(st.get("mean_aspect_ratio", np.nan)))
        count_avg.append(float(st.get("count_avg", np.nan)))

    for entry in per_image:
        t = float(entry.get("time", np.nan))
        if np.isnan(t):
            continue
        stats = entry.get("stats") or {}
        lengths_by_time.setdefault(t, []).extend(stats.get("lengths") or [])
        widths_by_time.setdefault(t, []).extend(stats.get("widths") or [])

    lengths_by_time = {k: np.asarray(v, dtype=float) for k, v in lengths_by_time.items()}
    widths_by_time = {k: np.asarray(v, dtype=float) for k, v in widths_by_time.items()}

    meta = data.get("meta") or {}
    label = meta.get("dataset") or path.stem
    scale = meta.get("scale") or {}
    um_per_px_from_meta = scale.get("um_per_px")

    return GuiSeries(
        label=label,
        times=np.asarray(times, dtype=float),
        mean_length_px=np.asarray(mean_len, dtype=float),
        mean_width_px=np.asarray(mean_wid, dtype=float),
        mean_ar=np.asarray(mean_ar, dtype=float),
        count_avg=np.asarray(count_avg, dtype=float),
        lengths_by_time=lengths_by_time,
        widths_by_time=widths_by_time,
        um_per_px_from_meta=float(um_per_px_from_meta) if um_per_px_from_meta else None,
    )


def _match_times(gui_times: np.ndarray, ref_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each GUI time, pick the nearest reference time (within 0.5 min)."""
    pairs_g, pairs_r = [], []
    for gt in gui_times:
        idx = int(np.argmin(np.abs(ref_times - gt)))
        if abs(ref_times[idx] - gt) <= 0.51:
            pairs_g.append(gt)
            pairs_r.append(ref_times[idx])
    return np.asarray(pairs_g, dtype=float), np.asarray(pairs_r, dtype=float)


def fit_um_per_px(gui: GuiSeries, ref: SupervisorSeries) -> float:
    gui_t, ref_t = _match_times(gui.times, ref.times)
    if len(gui_t) < 2:
        raise ValueError("Need at least two overlapping time points to fit scale")

    gui_means, ref_means = [], []
    gui_map = {float(t): v for t, v in zip(gui.times, gui.mean_length_px)}
    ref_map = {float(t): v for t, v in zip(ref.times, ref.mean_um)}
    for gt, rt in zip(gui_t, ref_t):
        g = gui_map.get(float(gt), np.nan)
        r = ref_map.get(float(rt), np.nan)
        if g > 0 and r > 0 and np.isfinite(g) and np.isfinite(r):
            gui_means.append(g)
            ref_means.append(r)

    if len(gui_means) < 2:
        raise ValueError("Not enough valid mean-length pairs for scale fit")
    scales = np.asarray(ref_means) / np.asarray(gui_means)
    return float(np.median(scales))


def length_weighted_histogram(lengths_um: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    lengths_um = lengths_um[np.isfinite(lengths_um) & (lengths_um > 0)]
    if lengths_um.size == 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    hist, _ = np.histogram(lengths_um, bins=bin_edges, weights=lengths_um)
    return hist.astype(float)


def _hist_l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (a.sum() or 1.0)
    b = b / (b.sum() or 1.0)
    return float(np.abs(a - b).sum())


def _nearest_ref_index(ref_times: np.ndarray, t: float) -> Optional[int]:
    idx = int(np.argmin(np.abs(ref_times - t)))
    return idx if abs(ref_times[idx] - t) <= 0.51 else None


def build_comparison_table(gui: GuiSeries, refs: SupervisorComparison, um_per_px: float) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(gui.times):
        row = {
            "time_min": t,
            "gui_count": gui.count_avg[i],
            "gui_mean_len_px": gui.mean_length_px[i],
            "gui_mean_len_um": gui.mean_length_px[i] * um_per_px,
            "gui_mean_width_um": gui.mean_width_px[i] * um_per_px,
            "gui_mean_ar": gui.mean_ar[i],
        }
        for key, ref in (
            ("length_weighted", refs.length_weighted),
            ("square_weighted", refs.square_weighted),
        ):
            ridx = _nearest_ref_index(ref.times, float(t))
            if ridx is None:
                continue
            rt = float(ref.times[ridx])
            prefix = key
            row[f"ref_{prefix}_time_min"] = rt
            row[f"ref_{prefix}_mean_um"] = ref.mean_um[ridx]
            row[f"ref_{prefix}_count"] = ref.count[ridx]
            row[f"delta_{prefix}_mean_um"] = row["gui_mean_len_um"] - ref.mean_um[ridx]
            row[f"delta_{prefix}_mean_pct"] = (
                100.0 * (row["gui_mean_len_um"] - ref.mean_um[ridx]) / ref.mean_um[ridx]
                if ref.mean_um[ridx]
                else np.nan
            )
            row[f"delta_{prefix}_count"] = row["gui_count"] - ref.count[ridx]
            if key == "length_weighted":
                gui_lens = gui.lengths_by_time.get(float(t))
                if gui_lens is not None and gui_lens.size:
                    gui_hist = length_weighted_histogram(gui_lens * um_per_px, ref.bin_edges_um)
                    ref_hist = ref.hist_by_time[ridx]
                    row["hist_l1_distance_length_weighted"] = _hist_l1_distance(gui_hist, ref_hist)
        rows.append(row)
    return pd.DataFrame(rows)


def _normalize_to_max(values: np.ndarray) -> np.ndarray:
    """Scale a series to [0, 1] using its own maximum."""
    arr = np.asarray(values, dtype=float)
    peak = np.nanmax(arr)
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(arr, dtype=float)
    return arr / peak


def plot_time_series(gui: GuiSeries, refs: SupervisorComparison, um_per_px: float, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"PSD comparison — {gui.label}\n"
        f"{refs.length_weighted.label} & {refs.square_weighted.label}",
        fontsize=12,
    )

    ax = axes[0, 0]
    ax.plot(gui.times, gui.mean_length_px * um_per_px, "o-", label="crystalGUI", color="#5b9cff", linewidth=2)
    ax.plot(
        refs.length_weighted.times, refs.length_weighted.mean_um, "s--",
        label=refs.length_weighted.label, color="#f59e0b", linewidth=2,
    )
    ax.plot(
        refs.square_weighted.times, refs.square_weighted.mean_um, "^--",
        label=refs.square_weighted.label, color="#a855f7", linewidth=2,
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mean size (µm)")
    ax.set_title("Mean crystal size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(gui.times, _normalize_to_max(gui.count_avg), "o-", label="crystalGUI", color="#5b9cff", linewidth=2)
    ax.plot(
        refs.length_weighted.times, _normalize_to_max(refs.length_weighted.count), "s--",
        label=refs.length_weighted.label, color="#f59e0b", linewidth=2,
    )
    ax.plot(
        refs.square_weighted.times, _normalize_to_max(refs.square_weighted.count), "^--",
        label=refs.square_weighted.label, color="#a855f7", linewidth=2,
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Count / max count")
    ax.set_title("Crystal count (normalized to max)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(gui.times, gui.mean_ar, "o-", color="#10b981")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Aspect ratio (L/W)")
    ax.set_title("Mean aspect ratio (GUI only)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    table = build_comparison_table(gui, refs, um_per_px)
    matched_len = table.dropna(subset=["delta_length_weighted_mean_pct"])
    matched_sq = table.dropna(subset=["delta_square_weighted_mean_pct"])
    if not matched_len.empty or not matched_sq.empty:
        width = 0.35
        x = np.arange(max(len(matched_len), len(matched_sq)))
        if not matched_len.empty:
            ax.bar(
                matched_len["time_min"] - width / 2,
                matched_len["delta_length_weighted_mean_pct"],
                width=width, label="vs length-weighted", color="#f59e0b", alpha=0.85,
            )
        if not matched_sq.empty:
            ax.bar(
                matched_sq["time_min"] + width / 2,
                matched_sq["delta_square_weighted_mean_pct"],
                width=width, label="vs square-weighted", color="#a855f7", alpha=0.85,
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Δ mean size (%)")
        ax.set_title("GUI − supervisor mean size (%)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No overlapping times", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "comparison_timeseries.png", dpi=160)
    plt.close(fig)


def _normalize_density(hist: np.ndarray) -> np.ndarray:
    """Normalize a histogram to unit sum so total crystal count does not affect the plot."""
    total = float(hist.sum())
    if total <= 0:
        return np.zeros_like(hist, dtype=float)
    return hist.astype(float) / total


def _plot_psd_overlay_figure(
    *,
    t: float,
    centers: np.ndarray,
    gui_hist: np.ndarray,
    ref_len_hist: Optional[np.ndarray],
    ref_len_t: Optional[float],
    ref_sq_hist: Optional[np.ndarray],
    ref_sq_t: Optional[float],
    ref_len_label: str,
    ref_sq_label: str,
    ylabel: str,
    title_suffix: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    x = np.arange(len(centers))
    offset = -1.5 * width
    if ref_len_hist is not None and ref_len_t is not None:
        ax.bar(
            x + offset, ref_len_hist, width=width,
            label=f"{ref_len_label} t={ref_len_t:g} min", color="#f59e0b", alpha=0.85,
        )
        offset += width
    if ref_sq_hist is not None and ref_sq_t is not None:
        ax.bar(
            x + offset, ref_sq_hist, width=width,
            label=f"{ref_sq_label} t={ref_sq_t:g} min", color="#a855f7", alpha=0.85,
        )
        offset += width
    ax.bar(
        x + offset, gui_hist, width=width,
        label=f"crystalGUI t={t:g} min", color="#5b9cff", alpha=0.85,
    )
    ax.set_xlabel("Length bin (µm)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"PSD at t≈{t:g} min{title_suffix}")
    step = max(1, len(centers) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([f"{centers[i]:.1f}" for i in range(0, len(centers), step)], rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_psd_overlays(
    gui: GuiSeries,
    refs: SupervisorComparison,
    um_per_px: float,
    out_dir: Path,
    times: Optional[List[float]] = None,
) -> None:
    if times is None:
        times = [float(t) for t in gui.times]

    ref_len = refs.length_weighted
    ref_sq = refs.square_weighted
    centers = np.sqrt(ref_len.bin_edges_um[:-1] * ref_len.bin_edges_um[1:])

    weighted_dir = out_dir / "psd_weighted"
    density_dir = out_dir / "psd_density"

    for t in times:
        ridx_len = _nearest_ref_index(ref_len.times, float(t))
        ridx_sq = _nearest_ref_index(ref_sq.times, float(t))
        gui_lens = gui.lengths_by_time.get(float(t))
        if gui_lens is None or gui_lens.size == 0:
            continue
        if ridx_len is None and ridx_sq is None:
            continue

        gui_hist = length_weighted_histogram(gui_lens * um_per_px, ref_len.bin_edges_um)
        ref_len_hist = ref_len.hist_by_time[ridx_len] if ridx_len is not None else None
        ref_sq_hist = ref_sq.hist_by_time[ridx_sq] if ridx_sq is not None else None
        ref_len_t = float(ref_len.times[ridx_len]) if ridx_len is not None else None
        ref_sq_t = float(ref_sq.times[ridx_sq]) if ridx_sq is not None else None

        safe_t = re.sub(r"[^0-9.]+", "_", f"{t:g}")
        out_name = f"comparison_psd_t{safe_t}min.png"

        _plot_psd_overlay_figure(
            t=t,
            centers=centers,
            gui_hist=gui_hist,
            ref_len_hist=ref_len_hist,
            ref_len_t=ref_len_t,
            ref_sq_hist=ref_sq_hist,
            ref_sq_t=ref_sq_t,
            ref_len_label=ref_len.label,
            ref_sq_label=ref_sq.label,
            ylabel="Weighted count",
            title_suffix="",
            out_path=weighted_dir / out_name,
        )
        _plot_psd_overlay_figure(
            t=t,
            centers=centers,
            gui_hist=_normalize_density(gui_hist),
            ref_len_hist=_normalize_density(ref_len_hist) if ref_len_hist is not None else None,
            ref_len_t=ref_len_t,
            ref_sq_hist=_normalize_density(ref_sq_hist) if ref_sq_hist is not None else None,
            ref_sq_t=ref_sq_t,
            ref_len_label=ref_len.label,
            ref_sq_label=ref_sq.label,
            ylabel="Density (normalized)",
            title_suffix=" — shape only",
            out_path=density_dir / out_name,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gui", required=True, type=Path, help="Full JSON export from Outputs tab")
    parser.add_argument("--excel", required=True, type=Path, help="Supervisor Excel workbook")
    parser.add_argument("--sheet", required=True, help="Excel sheet name (e.g. 'Reference-2.5mgmL, 20C, 400RPM')")
    parser.add_argument(
        "--table",
        default="auto",
        choices=["auto", "blaze", "microscopy"],
        help="Which time-series block inside the sheet to use (default: auto)",
    )
    parser.add_argument("--um-per-px", type=float, default=None, help="Microns per pixel for GUI lengths")
    parser.add_argument("--fit-scale", action="store_true", help="Estimate um-per-px from overlapping mean lengths")
    parser.add_argument("--out", type=Path, default=Path("data/psd_comparison"), help="Output directory")
    parser.add_argument("--times", type=str, default="", help="Comma-separated times for PSD overlays (default: all GUI times)")
    args = parser.parse_args()

    gui = load_gui_export(args.gui)
    refs = load_supervisor_excel(args.excel, args.sheet, table=args.table)

    um_per_px = args.um_per_px
    if um_per_px is None and gui.um_per_px_from_meta:
        um_per_px = gui.um_per_px_from_meta
        print(f"[json scale] Using um_per_px = {um_per_px:.6f} from export metadata")
    if um_per_px is None:
        if not args.fit_scale:
            parser.error("Provide --um-per-px, use --fit-scale, or export JSON with scale calibration")
        um_per_px = fit_um_per_px(gui, refs.length_weighted)
        print(f"[fit-scale] Estimated um_per_px = {um_per_px:.6f} (vs length-weighted)")
    else:
        print(f"Using um_per_px = {um_per_px:.6f}")

    args.out.mkdir(parents=True, exist_ok=True)

    table = build_comparison_table(gui, refs, um_per_px)
    table.to_csv(args.out / "comparison_table.csv", index=False)

    with (args.out / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "gui_label": gui.label,
                "ref_length_weighted_label": refs.length_weighted.label,
                "ref_square_weighted_label": refs.square_weighted.label,
                "sheet": args.sheet,
                "um_per_px": um_per_px,
                "matched_points": int(table["ref_length_weighted_mean_um"].notna().sum()),
                "mean_abs_delta_mean_pct_length_weighted": float(
                    table["delta_length_weighted_mean_pct"].abs().mean(skipna=True)
                )
                if table["delta_length_weighted_mean_pct"].notna().any()
                else None,
                "mean_abs_delta_mean_pct_square_weighted": float(
                    table["delta_square_weighted_mean_pct"].abs().mean(skipna=True)
                )
                if table["delta_square_weighted_mean_pct"].notna().any()
                else None,
                "mean_hist_l1_length_weighted": float(
                    table["hist_l1_distance_length_weighted"].mean(skipna=True)
                )
                if table["hist_l1_distance_length_weighted"].notna().any()
                else None,
            },
            f,
            indent=2,
        )

    plot_time_series(gui, refs, um_per_px, args.out)

    overlay_times = [float(x.strip()) for x in args.times.split(",") if x.strip()] or None
    plot_psd_overlays(gui, refs, um_per_px, args.out, times=overlay_times)

    print(f"\nWrote comparison outputs to {args.out.resolve()}")
    print("  comparison_table.csv")
    print("  comparison_summary.json")
    print("  comparison_timeseries.png")
    print("  psd_weighted/comparison_psd_t<time>min.png")
    print("  psd_density/comparison_psd_t<time>min.png")
    if not table.empty:
        show_cols = [
            "time_min", "gui_mean_len_um",
            "ref_length_weighted_mean_um", "delta_length_weighted_mean_pct",
            "ref_square_weighted_mean_um", "delta_square_weighted_mean_pct",
            "gui_count", "ref_length_weighted_count", "ref_square_weighted_count",
        ]
        show_cols = [c for c in show_cols if c in table.columns]
        print("\n" + table[show_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
