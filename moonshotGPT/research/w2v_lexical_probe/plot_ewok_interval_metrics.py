"""Plot periodic EWoK interval metrics produced by the Word2Vec lexical probe.

Example:
```bash
python -m research.w2v_lexical_probe.plot_ewok_interval_metrics \
  --interval_metrics runs/research/w2v_lexical_probe/<run_name>/ewok_interval_metrics.jsonl
```
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DEFAULT_METRIC_KEY = "domain_scores_full"
_PAIR_LABELS = ("official", "symmetric")
_Y_TICKS = tuple(step / 10.0 for step in range(11))
_PAIR_LINE_OPACITY = 0.4
_PAIR_LINE_WIDTH = 1.2
_PAIR_MARKER_RADIUS = 2.0
_MEAN_LINE_WIDTH = 2.6
_MEAN_COLOR = "#1f1f1f"
_BASELINE_MEAN_COLOR = "#c45c18"
_METRIC_DISPLAY_NAMES = {
    "domain_scores_full": "BabyLM Completion Choice",
    "domain_scores_official": "BabyLM Completion Choice",
    "ewok_context_sensitivity_domain_scores_full": "EWoK Context Sensitivity",
    "ewok_context_sensitivity_domain_scores_official": "EWoK Context Sensitivity",
}
_DEFAULT_COMPANION_METRIC_KEY = "ewok_context_sensitivity_domain_scores_full"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_") or "unknown"


def _metric_display_name(metric_key: str) -> str:
    return _METRIC_DISPLAY_NAMES.get(metric_key, metric_key)


def _figure_title(metric_key: str, baseline_label: str | None = None) -> str:
    title = f"Word2Vec EWoK Interval Metrics by Domain: {_metric_display_name(metric_key)}"
    if baseline_label:
        title += f" vs {baseline_label}"
    return title


def _load_interval_records(interval_metrics_path: str | Path) -> list[dict]:
    path = Path(interval_metrics_path).expanduser().resolve()
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _metric_payload_records(records: list[dict], metric_key: str) -> list[dict]:
    return [
        rec
        for rec in records
        if isinstance(rec, dict)
        and isinstance(rec.get("words_trained_total"), int)
        and isinstance(rec.get(metric_key), dict)
    ]


def _has_metric_payload(records: list[dict], metric_key: str) -> bool:
    return bool(_metric_payload_records(records, metric_key))


def _load_baseline_scores(baseline_metrics_path: str | Path, metric_key: str) -> dict | None:
    baseline_metrics_path = Path(baseline_metrics_path).expanduser().resolve()
    with baseline_metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    direct_payload = payload.get(metric_key)
    if isinstance(direct_payload, dict):
        return direct_payload

    metrics_by_method = payload.get("metrics_by_method", {})
    if metric_key == "domain_scores_full":
        nested = metrics_by_method.get("babylm_completion_choice", {}).get("domain_scores_full")
        if isinstance(nested, dict):
            return nested
    if metric_key == "domain_scores_official":
        nested = metrics_by_method.get("babylm_completion_choice", {}).get("domain_scores_official")
        if isinstance(nested, dict):
            return nested
    if metric_key == "ewok_context_sensitivity_domain_scores_full":
        nested = metrics_by_method.get("ewok_context_sensitivity", {}).get("domain_scores_full")
        if isinstance(nested, dict):
            return nested
    if metric_key == "ewok_context_sensitivity_domain_scores_official":
        nested = metrics_by_method.get("ewok_context_sensitivity", {}).get("domain_scores_official")
        if isinstance(nested, dict):
            return nested

    return None


def _domain_order(records: list[dict], metric_key: str, include_average: bool) -> list[str]:
    if not records:
        return []
    last_payload = records[-1].get(metric_key, {})
    if not isinstance(last_payload, dict):
        return []
    domains = [str(key) for key in last_payload.keys() if include_average or str(key) != "average"]
    return sorted(domains, key=lambda value: (value == "average", value))


def _subplot_grid(n_panels: int) -> tuple[int, int]:
    if n_panels <= 0:
        return 1, 1
    ncols = min(3, max(1, n_panels))
    nrows = int(math.ceil(n_panels / ncols))
    return nrows, ncols


def _x_tick_values(xs: list[float]) -> list[float]:
    if not xs:
        return []
    x_min = min(xs)
    x_max = max(xs)
    start = int(math.floor(x_min))
    end = int(math.floor(x_max))
    if end < start:
        end = start
    span = end - start
    if span <= 20:
        step = 1
    elif span <= 40:
        step = 2
    elif span <= 100:
        step = 5
    else:
        step = 10
    ticks = list(range(start, end + 1, step))
    if not ticks:
        ticks = [round(x_min, 1), round(x_max, 1)]
    elif ticks[-1] != end:
        ticks.append(end)
    return [float(value) for value in ticks]


def _xml_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_line(x1, y1, x2, y2, *, stroke="#000000", stroke_width=1.0, dash=None, opacity=None):
    attrs = [
        f'x1="{x1:.2f}"',
        f'y1="{y1:.2f}"',
        f'x2="{x2:.2f}"',
        f'y2="{y2:.2f}"',
        f'stroke="{stroke}"',
        f'stroke-width="{stroke_width:.2f}"',
        'fill="none"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity:.3f}"')
    return f"<line {' '.join(attrs)} />"


def _svg_polyline(points, *, stroke="#000000", stroke_width=1.8, opacity=None):
    if len(points) < 2:
        return ""
    attrs = [
        f'points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}"',
        f'stroke="{stroke}"',
        f'stroke-width="{stroke_width:.2f}"',
        'fill="none"',
        'stroke-linejoin="round"',
        'stroke-linecap="round"',
    ]
    if opacity is not None:
        attrs.append(f'opacity="{opacity:.3f}"')
    return f"<polyline {' '.join(attrs)} />"


def _svg_circle(x, y, radius=2.6, *, fill="#000000"):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" />'


def _svg_text(x, y, text, *, font_size=11, anchor="start", weight="normal", fill="#111111", rotation=None):
    attrs = [
        f'x="{x:.2f}"',
        f'y="{y:.2f}"',
        f'font-size="{font_size}"',
        f'text-anchor="{anchor}"',
        'font-family="sans-serif"',
        f'font-weight="{weight}"',
        f'fill="{fill}"',
    ]
    if rotation is not None:
        attrs.append(f'transform="rotate({rotation:.2f} {x:.2f} {y:.2f})"')
    return f"<text {' '.join(attrs)}>{_xml_escape(text)}</text>"


def _write_svg_interval_plot(
    records: list[dict],
    domains: list[str],
    metric_key: str,
    output_path: Path,
    baseline_scores: dict | None = None,
    baseline_label: str | None = None,
):
    nrows, ncols = _subplot_grid(len(domains))
    panel_w = 520
    panel_h = 320
    fig_w = panel_w * ncols
    fig_h = panel_h * nrows + 60
    outer_margin = 18
    title_y = 30

    global_xs = [rec["words_trained_total"] / 1_000_000_000.0 for rec in records]
    x_min = min(global_xs)
    x_max = max(global_xs)
    if x_max <= x_min:
        x_max = x_min + 1.0
    x_ticks = _x_tick_values(global_xs)

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig_w}" height="{fig_h}" viewBox="0 0 {fig_w} {fig_h}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white" />',
        _svg_text(
            fig_w / 2.0,
            title_y,
            _figure_title(metric_key, baseline_label=baseline_label),
            font_size=18,
            anchor="middle",
            weight="bold",
        ),
    ]

    colors = {
        "official": "#2a6f97",
        "symmetric": "#6c9f3a",
        "mean": _MEAN_COLOR,
        "baseline_mean": _BASELINE_MEAN_COLOR,
        "chance": "#d62728",
        "grid": "#d8d8d8",
        "axis": "#5b5b5b",
    }

    for idx, domain in enumerate(domains):
        row = idx // ncols
        col = idx % ncols
        panel_x = col * panel_w
        panel_y = row * panel_h + 50
        left = panel_x + outer_margin + 46
        top = panel_y + 30
        width = panel_w - 2 * outer_margin - 34
        height = panel_h - 2 * outer_margin - 56
        right = left + width
        bottom = top + height

        payload_points = []
        pair_mode = None
        for rec in records:
            payload = rec.get(metric_key, {})
            if not isinstance(payload, dict) or domain not in payload:
                continue
            x_words = rec["words_trained_total"] / 1_000_000_000.0
            value = payload[domain]
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                pair_mode = True
                payload_points.append((x_words, float(value[0]), float(value[1])))
            else:
                pair_mode = False
                payload_points.append((x_words, float(value)))

        chunks.append(f'<rect x="{panel_x:.2f}" y="{panel_y:.2f}" width="{panel_w - 2:.2f}" height="{panel_h - 8:.2f}" fill="white" stroke="#e5e5e5" stroke-width="1" />')
        chunks.append(_svg_text(panel_x + panel_w / 2.0, panel_y + 18, domain, font_size=12, anchor="middle", weight="bold"))

        for y_value in _Y_TICKS:
            y = bottom - y_value * height
            if y_value == 0.5:
                chunks.append(_svg_line(left, y, right, y, stroke=colors["chance"], stroke_width=1.1, dash="8 2 2 2"))
            else:
                chunks.append(_svg_line(left, y, right, y, stroke=colors["grid"], stroke_width=0.8))
            chunks.append(_svg_text(left - 8, y + 4, f"{y_value:.1f}", font_size=9, anchor="end", fill="#4f4f4f"))

        for x_value in x_ticks:
            frac = (x_value - x_min) / (x_max - x_min)
            x = left + frac * width
            chunks.append(_svg_line(x, bottom, x, top, stroke=colors["grid"], stroke_width=0.8, opacity=0.7))
            chunks.append(_svg_text(x, bottom + 26, f"{x_value:.0f}", font_size=8, anchor="end", fill="#4f4f4f", rotation=-45))

        chunks.append(_svg_line(left, bottom, right, bottom, stroke=colors["axis"], stroke_width=1.0))
        chunks.append(_svg_line(left, top, left, bottom, stroke=colors["axis"], stroke_width=1.0))
        chunks.append(_svg_text(left + width / 2.0, bottom + 34, "Words Trained (B)", font_size=9, anchor="middle"))
        chunks.append(_svg_text(left - 34, top + height / 2.0, "Accuracy", font_size=9, anchor="middle"))

        baseline_domain_value = baseline_scores.get(domain) if isinstance(baseline_scores, dict) else None
        baseline_mean_y = None
        if isinstance(baseline_domain_value, (list, tuple)) and len(baseline_domain_value) >= 2:
            baseline_mean_y = 0.5 * (float(baseline_domain_value[0]) + float(baseline_domain_value[1]))
        elif isinstance(baseline_domain_value, (float, int)):
            baseline_mean_y = float(baseline_domain_value)
        if baseline_mean_y is not None:
            baseline_y = bottom - baseline_mean_y * height
            chunks.append(
                _svg_line(
                    left,
                    baseline_y,
                    right,
                    baseline_y,
                    stroke=colors["baseline_mean"],
                    stroke_width=1.8,
                    dash="6 4",
                    opacity=0.95,
                )
            )

        if payload_points:
            if pair_mode:
                pts_official = []
                pts_symmetric = []
                pts_mean = []
                for x_words, y_official, y_symmetric in payload_points:
                    x = left + ((x_words - x_min) / (x_max - x_min)) * width
                    y0 = bottom - y_official * height
                    y1 = bottom - y_symmetric * height
                    ym = bottom - ((0.5 * (y_official + y_symmetric)) * height)
                    pts_official.append((x, y0))
                    pts_symmetric.append((x, y1))
                    pts_mean.append((x, ym))
                chunks.append(
                    _svg_polyline(
                        pts_official,
                        stroke=colors["official"],
                        stroke_width=_PAIR_LINE_WIDTH,
                        opacity=_PAIR_LINE_OPACITY,
                    )
                )
                chunks.append(
                    _svg_polyline(
                        pts_symmetric,
                        stroke=colors["symmetric"],
                        stroke_width=_PAIR_LINE_WIDTH,
                        opacity=_PAIR_LINE_OPACITY,
                    )
                )
                chunks.append(
                    _svg_polyline(
                        pts_mean,
                        stroke=colors["mean"],
                        stroke_width=_MEAN_LINE_WIDTH,
                        opacity=1.0,
                    )
                )
                for x, y in pts_official:
                    chunks.append(_svg_circle(x, y, radius=_PAIR_MARKER_RADIUS, fill=colors["official"]))
                for x, y in pts_symmetric:
                    chunks.append(_svg_circle(x, y, radius=_PAIR_MARKER_RADIUS, fill=colors["symmetric"]))
                for x, y in pts_mean:
                    chunks.append(_svg_circle(x, y, radius=2.4, fill=colors["mean"]))
                legend_items = [("mean", colors["mean"]), ("official", colors["official"]), ("symmetric", colors["symmetric"])]
            else:
                pts_scalar = []
                for x_words, y_value in payload_points:
                    x = left + ((x_words - x_min) / (x_max - x_min)) * width
                    y = bottom - y_value * height
                    pts_scalar.append((x, y))
                chunks.append(_svg_polyline(pts_scalar, stroke=colors["official"], stroke_width=1.8))
                for x, y in pts_scalar:
                    chunks.append(_svg_circle(x, y, fill=colors["official"]))
                legend_items = [(_metric_display_name(metric_key), colors["official"])]

            if baseline_mean_y is not None:
                legend_items.insert(1, (f"{baseline_label or 'baseline'} mean", colors["baseline_mean"]))

            legend_y = top + 12
            legend_x = right - 120
            for label, color in legend_items:
                dash = "6 4" if "baseline" in str(label) else None
                chunks.append(_svg_line(legend_x, legend_y - 4, legend_x + 16, legend_y - 4, stroke=color, stroke_width=2.0, dash=dash))
                chunks.append(_svg_text(legend_x + 22, legend_y, label, font_size=9))
                legend_y += 14

    chunks.append("</svg>")
    output_path.write_text("\n".join(chunks), encoding="utf-8")


def plot_ewok_interval_metrics(
    interval_metrics_path: str | Path,
    *,
    metric_key: str = DEFAULT_METRIC_KEY,
    output_path: str | Path | None = None,
    include_average: bool = True,
    baseline_scores: dict | None = None,
    baseline_label: str | None = None,
) -> Path:
    """Plot one subplot per domain from the periodic EWoK interval JSONL log."""
    interval_metrics_path = Path(interval_metrics_path).expanduser().resolve()
    records = _metric_payload_records(_load_interval_records(interval_metrics_path), metric_key)
    if not records:
        raise ValueError(f"No interval records with metric_key={metric_key!r} found in {interval_metrics_path}")

    domains = _domain_order(records, metric_key, include_average=include_average)
    if not domains:
        raise ValueError(f"No domains found for metric_key={metric_key!r} in {interval_metrics_path}")

    if output_path is None:
        suffix = ".png" if plt is not None else ".svg"
        output_path = interval_metrics_path.with_name(
            f"{interval_metrics_path.stem}_{_safe_name(metric_key)}_subplots{suffix}"
        )
    output_path = Path(output_path).expanduser().resolve()

    if plt is None:
        _write_svg_interval_plot(
            records,
            domains,
            metric_key,
            output_path,
            baseline_scores=baseline_scores,
            baseline_label=baseline_label,
        )
        return output_path

    nrows, ncols = _subplot_grid(len(domains))
    global_xs = [rec["words_trained_total"] / 1_000_000_000.0 for rec in records]
    x_ticks = _x_tick_values(global_xs)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, max(4.5, nrows * 3.8)),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, domain in enumerate(domains):
        ax = axes_flat[idx]
        xs = []
        ys_scalar = []
        ys_pair_0 = []
        ys_pair_1 = []
        pair_mode = None

        for rec in records:
            payload = rec.get(metric_key, {})
            if not isinstance(payload, dict) or domain not in payload:
                continue
            x_words = int(rec["words_trained_total"])
            value = payload[domain]
            xs.append(x_words / 1_000_000_000.0)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                pair_mode = True
                ys_pair_0.append(float(value[0]))
                ys_pair_1.append(float(value[1]))
            else:
                pair_mode = False
                ys_scalar.append(float(value))

        if pair_mode:
            ax.plot(
                xs,
                ys_pair_0,
                marker="o",
                linewidth=_PAIR_LINE_WIDTH,
                markersize=2.8,
                color="#2a6f97",
                alpha=_PAIR_LINE_OPACITY,
                label=_PAIR_LABELS[0],
                zorder=2,
            )
            ax.plot(
                xs,
                ys_pair_1,
                marker="o",
                linewidth=_PAIR_LINE_WIDTH,
                markersize=2.8,
                color="#6c9f3a",
                alpha=_PAIR_LINE_OPACITY,
                label=_PAIR_LABELS[1],
                zorder=2,
            )
            ys_mean = [(left + right) * 0.5 for left, right in zip(ys_pair_0, ys_pair_1)]
            ax.plot(
                xs,
                ys_mean,
                marker="o",
                linewidth=_MEAN_LINE_WIDTH,
                markersize=3.8,
                color=_MEAN_COLOR,
                alpha=1.0,
                label="mean",
                zorder=3,
            )
        else:
            ax.plot(xs, ys_scalar, marker="o", linewidth=1.8, markersize=3.5, color="#2a6f97", label=metric_key)

        baseline_domain_value = baseline_scores.get(domain) if isinstance(baseline_scores, dict) else None
        baseline_mean_y = None
        if isinstance(baseline_domain_value, (list, tuple)) and len(baseline_domain_value) >= 2:
            baseline_mean_y = 0.5 * (float(baseline_domain_value[0]) + float(baseline_domain_value[1]))
        elif isinstance(baseline_domain_value, (float, int)):
            baseline_mean_y = float(baseline_domain_value)
        if baseline_mean_y is not None:
            ax.axhline(
                baseline_mean_y,
                color=_BASELINE_MEAN_COLOR,
                linestyle=(0, (6, 4)),
                linewidth=1.8,
                label=f"{baseline_label or 'baseline'} mean",
                zorder=1,
            )

        ax.axhline(0.5, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.0, label="chance = 50%")
        ax.set_title(domain, fontsize=10)
        ax.set_xlabel("Words Trained (B)", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(_Y_TICKS)
        if x_ticks:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{tick:.0f}" for tick in x_ticks], rotation=45, ha="right")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for idx in range(len(domains), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(_figure_title(metric_key, baseline_label=baseline_label), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _companion_output_path(primary_output_path: Path, companion_metric_key: str) -> Path:
    suffix_to_strip = f"_{_safe_name(DEFAULT_METRIC_KEY)}_subplots"
    stem = primary_output_path.stem
    if stem.endswith(suffix_to_strip):
        base_stem = stem[: -len(suffix_to_strip)]
    else:
        base_stem = stem
    return primary_output_path.with_name(
        f"{base_stem}_{_safe_name(companion_metric_key)}_subplots{primary_output_path.suffix}"
    )


def plot_ewok_interval_metric_suite(
    interval_metrics_path: str | Path,
    *,
    primary_metric_key: str = DEFAULT_METRIC_KEY,
    output_path: str | Path | None = None,
    include_average: bool = True,
    write_companion_plot: bool = True,
    baseline_metrics: str | Path | None = None,
    baseline_label: str | None = None,
) -> dict[str, Path]:
    interval_metrics_path = Path(interval_metrics_path).expanduser().resolve()
    raw_records = _load_interval_records(interval_metrics_path)
    primary_baseline_scores = _load_baseline_scores(baseline_metrics, primary_metric_key) if baseline_metrics is not None else None
    primary_path = plot_ewok_interval_metrics(
        interval_metrics_path,
        metric_key=primary_metric_key,
        output_path=output_path,
        include_average=include_average,
        baseline_scores=primary_baseline_scores,
        baseline_label=baseline_label,
    )
    output_paths = {primary_metric_key: primary_path}

    should_write_companion = (
        write_companion_plot
        and primary_metric_key == DEFAULT_METRIC_KEY
        and _DEFAULT_COMPANION_METRIC_KEY != primary_metric_key
        and _has_metric_payload(raw_records, _DEFAULT_COMPANION_METRIC_KEY)
    )
    if should_write_companion:
        companion_path = _companion_output_path(primary_path, _DEFAULT_COMPANION_METRIC_KEY)
        companion_baseline_scores = (
            _load_baseline_scores(baseline_metrics, _DEFAULT_COMPANION_METRIC_KEY)
            if baseline_metrics is not None
            else None
        )
        output_paths[_DEFAULT_COMPANION_METRIC_KEY] = plot_ewok_interval_metrics(
            interval_metrics_path,
            metric_key=_DEFAULT_COMPANION_METRIC_KEY,
            output_path=companion_path,
            include_average=include_average,
            baseline_scores=companion_baseline_scores,
            baseline_label=baseline_label,
        )

    return output_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot EWoK interval metrics for a Word2Vec lexical-probe run.")
    parser.add_argument("--interval_metrics", required=True, help="Path to ewok_interval_metrics.jsonl")
    parser.add_argument(
        "--metric_key",
        default=DEFAULT_METRIC_KEY,
        help="Top-level metric payload to plot from each JSONL record (default: domain_scores_full)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Optional PNG output path. Defaults next to the JSONL file.",
    )
    parser.add_argument(
        "--include_average",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include the macro-average subplot",
    )
    parser.add_argument(
        "--write_companion_plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When plotting the default BabyLM metric, also emit the EWoK context-sensitivity plot",
    )
    parser.add_argument(
        "--baseline_metrics",
        default=None,
        help="Optional ewok_metrics.json path whose domain scores should be drawn as horizontal baseline lines",
    )
    parser.add_argument(
        "--baseline_label",
        default="Google News Word2Vec",
        help="Legend/title label for the optional baseline overlay",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    output_paths = plot_ewok_interval_metric_suite(
        args.interval_metrics,
        primary_metric_key=args.metric_key,
        output_path=args.output_path,
        include_average=args.include_average,
        write_companion_plot=args.write_companion_plot,
        baseline_metrics=args.baseline_metrics,
        baseline_label=args.baseline_label,
    )
    print(
        json.dumps(
            {
                "plot_paths": {metric_key: str(path) for metric_key, path in output_paths.items()},
            },
            indent=2,
        )
    )
