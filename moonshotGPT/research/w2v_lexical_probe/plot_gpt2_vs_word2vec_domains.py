"""Compare final GPT-2 and Word2Vec EWoK per-domain scores in a 4x3 SVG plot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parents[2]
_W2V_RUNS_DIR = _PROJECT_DIR / "runs" / "research" / "w2v_lexical_probe"
_GPT_RUNS_DIR = _PROJECT_DIR / "runs" / "research" / "bos_aligned_proto"
DEFAULT_WORD2VEC_LABEL = "Word2Vec FineWeb-Edu"
DEFAULT_GPT2_LABEL = "GPT-2 Medium"
DEFAULT_METRIC_KEY = "domain_scores_full"
DEFAULT_GPT2_METRIC_KEY = "eval_full_mean"
_Y_TICKS = tuple(step / 10.0 for step in range(11))
_DOMAIN_ORDER = (
    "agent-properties",
    "material-dynamics",
    "material-properties",
    "physical-dynamics",
    "physical-interactions",
    "physical-relations",
    "quantitative-properties",
    "social-interactions",
    "social-properties",
    "social-relations",
    "spatial-relations",
    "average",
)


def _discover_default_path(env_key: str, search_dir: Path, pattern: str, fallback: Path) -> Path:
    env_path = os.environ.get(env_key)
    if env_path:
        return Path(env_path).expanduser()

    candidates = sorted(search_dir.glob(pattern))
    if len(candidates) == 1:
        return candidates[0]
    return fallback


DEFAULT_WORD2VEC_INTERVAL_METRICS = _discover_default_path(
    "MOONSHOT_WORD2VEC_INTERVAL_METRICS",
    _W2V_RUNS_DIR,
    "*/ewok_interval_metrics.jsonl",
    _W2V_RUNS_DIR / "<run_name>" / "ewok_interval_metrics.jsonl",
)
DEFAULT_GPT2_MEDIUM_METRICS = _discover_default_path(
    "MOONSHOT_GPT2_STEP_METRICS",
    _GPT_RUNS_DIR,
    "*/step_metrics.json",
    _GPT_RUNS_DIR / "<run_name>" / "step_metrics.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--word2vec-interval-metrics",
        type=Path,
        default=DEFAULT_WORD2VEC_INTERVAL_METRICS,
        help="JSONL file with periodic Word2Vec interval metrics.",
    )
    parser.add_argument(
        "--gpt2-medium-metrics",
        type=Path,
        default=DEFAULT_GPT2_MEDIUM_METRICS,
        help="JSON file with GPT-2 Medium step metrics.",
    )
    parser.add_argument(
        "--word2vec-label",
        default=DEFAULT_WORD2VEC_LABEL,
        help="Legend label for the Word2Vec run.",
    )
    parser.add_argument(
        "--gpt2-label",
        default=DEFAULT_GPT2_LABEL,
        help="Legend label for the GPT-2 run.",
    )
    parser.add_argument(
        "--metric-key",
        default=DEFAULT_METRIC_KEY,
        help="Word2Vec metric payload to compare (default: domain_scores_full).",
    )
    parser.add_argument(
        "--gpt2-metric-key",
        default=DEFAULT_GPT2_METRIC_KEY,
        help="GPT-2 metric payload to compare (default: eval_full_mean).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional SVG output path.",
    )
    return parser.parse_args()


def _pair_to_mean(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        left, right = value[0], value[1]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return 0.5 * (float(left) + float(right))
    return None


def _xml_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_text(x, y, text, *, font_size=11, anchor="start", weight="normal", fill="#111111"):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{font_size}" text-anchor="{anchor}" '
        f'font-family="sans-serif" font-weight="{weight}" fill="{fill}">{_xml_escape(text)}</text>'
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


def _svg_rect(x, y, width, height, *, fill="#ffffff", stroke="#000000", stroke_width=1.0, opacity=None):
    attrs = [
        f'x="{x:.2f}"',
        f'y="{y:.2f}"',
        f'width="{width:.2f}"',
        f'height="{height:.2f}"',
        f'fill="{fill}"',
        f'stroke="{stroke}"',
        f'stroke-width="{stroke_width:.2f}"',
    ]
    if opacity is not None:
        attrs.append(f'opacity="{opacity:.3f}"')
    return f"<rect {' '.join(attrs)} />"


def _load_last_word2vec_record(path: Path, metric_key: str) -> dict:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and isinstance(payload.get(metric_key), dict):
                records.append(payload)
    if not records:
        raise ValueError(f"No Word2Vec records with {metric_key!r} found in {path}")
    return records[-1]


def _load_last_gpt_record(path: Path, metric_key: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    records = [rec for rec in payload if isinstance(rec, dict) and isinstance(rec.get(metric_key), dict)]
    if not records:
        raise ValueError(f"No GPT-2 records with {metric_key!r} found in {path}")
    return records[-1]


def _extract_scalar_scores(payload: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    for domain in _DOMAIN_ORDER:
        value = payload.get(domain)
        scalar = _pair_to_mean(value)
        if scalar is not None:
            scores[domain] = scalar
    return scores


def _default_output_path(word2vec_interval_metrics: Path) -> Path:
    return word2vec_interval_metrics.with_name("gpt2_medium_vs_word2vec_final_babylm_domains_4x3.svg")


def write_svg_comparison(
    output_path: Path,
    *,
    gpt_scores: dict[str, float],
    word2vec_scores: dict[str, float],
    gpt_label: str,
    word2vec_label: str,
    gpt_step: int | None,
    word2vec_eval_index: int | None,
) -> None:
    ncols = 3
    nrows = 4
    panel_w = 500
    panel_h = 280
    fig_w = panel_w * ncols
    fig_h = panel_h * nrows + 92
    outer_margin = 20
    colors = {
        "gpt": "#2a6f97",
        "w2v": "#c45c18",
        "grid": "#d8d8d8",
        "axis": "#5b5b5b",
        "chance": "#d62728",
        "delta_pos": "#1f7a1f",
        "delta_neg": "#a11d21",
    }

    title = "BabyLM Completion Choice by Domain: GPT-2 Medium vs final Word2Vec"
    subtitle = []
    if gpt_step is not None:
        subtitle.append(f"GPT final step {gpt_step}")
    if word2vec_eval_index is not None:
        subtitle.append(f"Word2Vec final eval {word2vec_eval_index}")

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig_w}" height="{fig_h}" viewBox="0 0 {fig_w} {fig_h}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white" />',
        _svg_text(fig_w / 2.0, 28, title, font_size=20, anchor="middle", weight="bold"),
    ]
    if subtitle:
        chunks.append(_svg_text(fig_w / 2.0, 48, " | ".join(subtitle), font_size=11, anchor="middle", fill="#555555"))

    legend_x = fig_w / 2.0 - 180
    legend_y = 68
    chunks.append(_svg_rect(legend_x, legend_y - 10, 14, 14, fill=colors["gpt"], stroke=colors["gpt"]))
    chunks.append(_svg_text(legend_x + 22, legend_y + 2, gpt_label, font_size=10))
    chunks.append(_svg_rect(legend_x + 170, legend_y - 10, 14, 14, fill=colors["w2v"], stroke=colors["w2v"]))
    chunks.append(_svg_text(legend_x + 192, legend_y + 2, word2vec_label, font_size=10))

    ordered_domains = [domain for domain in _DOMAIN_ORDER if domain in gpt_scores and domain in word2vec_scores]
    for idx, domain in enumerate(ordered_domains):
        row = idx // ncols
        col = idx % ncols
        panel_x = col * panel_w
        panel_y = row * panel_h + 88
        left = panel_x + outer_margin + 44
        top = panel_y + 32
        width = panel_w - 2 * outer_margin - 32
        height = panel_h - 2 * outer_margin - 72
        right = left + width
        bottom = top + height

        chunks.append(_svg_rect(panel_x + 2, panel_y, panel_w - 6, panel_h - 10, fill="#ffffff", stroke="#e5e5e5"))
        chunks.append(_svg_text(panel_x + panel_w / 2.0, panel_y + 18, domain, font_size=12, anchor="middle", weight="bold"))

        for y_value in _Y_TICKS:
            y = bottom - y_value * height
            if y_value == 0.5:
                chunks.append(_svg_line(left, y, right, y, stroke=colors["chance"], stroke_width=1.0, dash="8 2 2 2"))
            else:
                chunks.append(_svg_line(left, y, right, y, stroke=colors["grid"], stroke_width=0.8))
            chunks.append(_svg_text(left - 8, y + 4, f"{y_value:.1f}", font_size=9, anchor="end", fill="#4f4f4f"))

        chunks.append(_svg_line(left, bottom, right, bottom, stroke=colors["axis"]))
        chunks.append(_svg_line(left, top, left, bottom, stroke=colors["axis"]))
        chunks.append(_svg_text(left - 32, top + height / 2.0, "Accuracy", font_size=9, anchor="middle"))

        bar_width = width * 0.22
        gap = width * 0.10
        total_bar_width = 2 * bar_width + gap
        bar_start = left + (width - total_bar_width) / 2.0

        gpt_score = gpt_scores[domain]
        w2v_score = word2vec_scores[domain]
        gpt_bar_h = gpt_score * height
        w2v_bar_h = w2v_score * height

        chunks.append(
            _svg_rect(
                bar_start,
                bottom - gpt_bar_h,
                bar_width,
                gpt_bar_h,
                fill=colors["gpt"],
                stroke=colors["gpt"],
                opacity=0.88,
            )
        )
        chunks.append(
            _svg_rect(
                bar_start + bar_width + gap,
                bottom - w2v_bar_h,
                bar_width,
                w2v_bar_h,
                fill=colors["w2v"],
                stroke=colors["w2v"],
                opacity=0.88,
            )
        )

        chunks.append(_svg_text(bar_start + bar_width / 2.0, bottom + 18, "GPT", font_size=9, anchor="middle"))
        chunks.append(
            _svg_text(
                bar_start + bar_width + gap + bar_width / 2.0,
                bottom + 18,
                "w2v",
                font_size=9,
                anchor="middle",
            )
        )
        chunks.append(_svg_text(bar_start + bar_width / 2.0, bottom - gpt_bar_h - 6, f"{gpt_score:.3f}", font_size=9, anchor="middle"))
        chunks.append(
            _svg_text(
                bar_start + bar_width + gap + bar_width / 2.0,
                bottom - w2v_bar_h - 6,
                f"{w2v_score:.3f}",
                font_size=9,
                anchor="middle",
            )
        )

        delta = gpt_score - w2v_score
        delta_fill = colors["delta_pos"] if delta >= 0 else colors["delta_neg"]
        chunks.append(
            _svg_text(
                panel_x + panel_w / 2.0,
                bottom + 38,
                f"GPT - w2v: {delta:+.3f}",
                font_size=10,
                anchor="middle",
                weight="bold",
                fill=delta_fill,
            )
        )

    chunks.append("</svg>")
    output_path.write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    args = parse_args()
    word2vec_interval_metrics = args.word2vec_interval_metrics.expanduser().resolve()
    gpt2_metrics = args.gpt2_medium_metrics.expanduser().resolve()

    word2vec_record = _load_last_word2vec_record(word2vec_interval_metrics, args.metric_key)
    gpt2_record = _load_last_gpt_record(gpt2_metrics, args.gpt2_metric_key)

    word2vec_scores = _extract_scalar_scores(word2vec_record[args.metric_key])
    gpt2_scores = _extract_scalar_scores(gpt2_record[args.gpt2_metric_key])

    output_path = args.output_path.expanduser().resolve() if args.output_path else _default_output_path(word2vec_interval_metrics)
    write_svg_comparison(
        output_path,
        gpt_scores=gpt2_scores,
        word2vec_scores=word2vec_scores,
        gpt_label=args.gpt2_label,
        word2vec_label=args.word2vec_label,
        gpt_step=gpt2_record.get("step") if isinstance(gpt2_record.get("step"), int) else None,
        word2vec_eval_index=word2vec_record.get("eval_index") if isinstance(word2vec_record.get("eval_index"), int) else None,
    )

    macro_gpt = gpt2_scores.get("average")
    macro_w2v = word2vec_scores.get("average")
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "gpt2_step": gpt2_record.get("step"),
                "word2vec_eval_index": word2vec_record.get("eval_index"),
                "gpt2_macro_average": macro_gpt,
                "word2vec_macro_average": macro_w2v,
                "macro_delta_gpt_minus_word2vec": (
                    None if macro_gpt is None or macro_w2v is None else macro_gpt - macro_w2v
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
