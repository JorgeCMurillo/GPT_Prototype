#!/usr/bin/env bash
set -euo pipefail

# Run the raw-window TrackStar material/social/physical query set with
# side-specific completion CE, then reproduce the lexical and strong-side
# audits used for the paired completion_ce run.
#
# Assumes the babylm environment is active. Override variables at invocation
# time if needed, for example:
#
#   NPROC=2 RAW_WINDOW_END_ID=250000 bash scripts/run_completion_side_ce_material_audit.sh

REPO_ROOT="${REPO_ROOT:-/home/jorge/tokenPred/moonshotGPT}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/experiments/babygpt_fineweb_bin_mbs4_T1024_d1024_h16_L24_tok491520_efftok491520_ws8_gas15_seed42_steps20000}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/processed/fineweb_edu_10B}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-20000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/ckpt_final_step0020000}"
FILTER_SPEC="${FILTER_SPEC:-${REPO_ROOT}/research/bos_aligned_proto/analysis/attribution/ewok_query_specs/domains_social_physical_material_dynamics.json}"

BASE_DIR="${BASE_DIR:-/SSD2/trackstar_raw_windows_250k_2p16_ckpt20000_social_physical_material_completion_side}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/results}"
# Reuse the paired run cache by default so candidate-side projected features do
# not need to be rebuilt if compatible cache entries already exist.
CACHE_DIR="${CACHE_DIR:-/SSD2/trackstar_raw_windows_250k_2p16_ckpt20000_social_physical_material/cache}"

RAW_WINDOW_START_ID="${RAW_WINDOW_START_ID:-0}"
RAW_WINDOW_END_ID="${RAW_WINDOW_END_ID:-250000}"
MAX_CANDIDATE_ROWS="${MAX_CANDIDATE_ROWS:-250000}"
TOPK="${TOPK:-100}"
BOTTOMK="${BOTTOMK:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PAPER_BLOCK_FEATURES="${PAPER_BLOCK_FEATURES:-4096}"
SCORE_CANDIDATE_CHUNK_SIZE="${SCORE_CANDIDATE_CHUNK_SIZE:-4096}"
NPROC="${NPROC:-2}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

echo "[1/3] Running TrackStar completion_side_ce..."
"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC}" \
  -m research.bos_aligned_proto.analysis.attribution.run_trackstar \
  --run_dir "${RUN_DIR}" \
  --data_dir "${DATA_DIR}" \
  --checkpoint_steps "${CHECKPOINT_STEP}" \
  --candidate_strategy raw_window_range \
  --raw_window_start_id "${RAW_WINDOW_START_ID}" \
  --raw_window_end_id "${RAW_WINDOW_END_ID}" \
  --max_candidate_rows "${MAX_CANDIDATE_ROWS}" \
  --candidate_kind stream_window \
  --ewok_variant fast \
  --ewok_filter_spec "${FILTER_SPEC}" \
  --ewok_score_view babylm_completion_choice \
  --ewok_target_scope both \
  --score_reduction mean \
  --query_objective completion_side_ce \
  --topk "${TOPK}" \
  --bottomk "${BOTTOMK}" \
  --distributed ddp \
  --batch_size "${BATCH_SIZE}" \
  --projection_layout paper_blocks \
  --paper_block_features "${PAPER_BLOCK_FEATURES}" \
  --score_candidate_chunk_size "${SCORE_CANDIDATE_CHUNK_SIZE}" \
  --output_dir "${OUT_DIR}" \
  --cache_dir "${CACHE_DIR}"

LEX_DIR="${OUT_DIR}/material_dynamics_completion_side_lexical_audit_step$(printf '%08d' "${CHECKPOINT_STEP}")"

echo "[2/3] Running side-aware lexical audit..."
"${PYTHON_BIN}" -m research.bos_aligned_proto.analysis.attribution.trackstar.run_lexical_audit \
  --attribution_dir "${OUT_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --output_dir "${LEX_DIR}" \
  --step "${CHECKPOINT_STEP}" \
  --domain material-dynamics \
  --topk "${TOPK}" \
  --randomk 100 \
  --max_tokens 1025 \
  --preview_chars 800

echo "[3/3] Building correct-side and strong-side summaries..."
ATTRIBUTION_DIR="${OUT_DIR}" LEX_DIR="${LEX_DIR}" STEP="${CHECKPOINT_STEP}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def step_tag(step: int) -> str:
    return f"step{int(step):08d}"


def side_from_target_id(target_id: str) -> str:
    marker = ":completion_side:"
    if marker not in str(target_id):
        return ""
    return str(target_id).rsplit(marker, 1)[1]


def base_target_id(target_id: str) -> str:
    marker = ":completion_side:"
    if marker not in str(target_id):
        return str(target_id)
    return str(target_id).rsplit(marker, 1)[0]


def boolify(frame: pd.DataFrame) -> pd.DataFrame:
    for col in (
        "has_query_exact_hit",
        "has_query_stem_hit",
        "has_concept_stem_hit",
        "has_material_keyword_hit",
    ):
        if col in frame and frame[col].dtype != bool:
            frame[col] = frame[col].astype(str).str.lower().isin({"true", "1", "yes"})
    return frame


def summarize(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "rows": int(len(frame)),
            "targets": int(frame["target_id"].nunique()) if len(frame) else 0,
            "mean_score": float(frame["score"].mean()) if len(frame) and frame["score"].notna().any() else np.nan,
            "any_query_exact_rate": float(frame["has_query_exact_hit"].mean()) if len(frame) else np.nan,
            "any_query_stem_rate": float(frame["has_query_stem_hit"].mean()) if len(frame) else np.nan,
            "any_concept_stem_rate": float(frame["has_concept_stem_hit"].mean()) if len(frame) else np.nan,
            "any_material_keyword_rate": float(frame["has_material_keyword_hit"].mean()) if len(frame) else np.nan,
            "mean_query_exact_fraction": float(frame["query_exact_overlap_fraction"].mean()) if len(frame) else np.nan,
            "mean_query_stem_fraction": float(frame["query_stem_overlap_fraction"].mean()) if len(frame) else np.nan,
            "mean_material_keyword_hit_count": float(frame["material_keyword_hit_count"].mean()) if len(frame) else np.nan,
            "median_material_keyword_hit_count": float(frame["material_keyword_hit_count"].median()) if len(frame) else np.nan,
        }
    )


def md_table(frame: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in frame.head(n).to_dict("records"):
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


step = int(os.environ["STEP"])
root = Path(os.environ["ATTRIBUTION_DIR"]).expanduser().resolve()
lex_dir = Path(os.environ["LEX_DIR"]).expanduser().resolve()
out_dir = root / f"material_dynamics_completion_side_strong_audit_{step_tag(step)}"
out_dir.mkdir(parents=True, exist_ok=True)

rows_path = lex_dir / f"material_dynamics_lexical_rows_{step_tag(step)}.csv"
rows = boolify(pd.read_csv(rows_path))

diagnostics = []
with (root / f"target_diagnostics_{step_tag(step)}.jsonl").open("r", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        if row.get("domain") == "material-dynamics":
            diagnostics.append(row)
diag = pd.DataFrame(diagnostics)
diag["completion_side"] = diag["target_id"].map(side_from_target_id)
diag["base_target_id"] = diag["target_id"].map(base_target_id)
diag["side_margin"] = np.where(diag["completion_side"].eq("c1_t1"), diag["margin_1"], diag["margin_2"])
diag["side_correct"] = diag["side_margin"] > 0
diag["strong_side_gt_0p2"] = diag["side_margin"] > 0.2
diag["very_strong_side_gt_0p5"] = diag["side_margin"] > 0.5

rows = rows.merge(
    diag[
        [
            "target_id",
            "base_target_id",
            "completion_side",
            "margin_1",
            "margin_2",
            "combined_margin",
            "side_margin",
            "side_correct",
            "strong_side_gt_0p2",
            "very_strong_side_gt_0p5",
        ]
    ],
    on=["target_id", "base_target_id", "completion_side"],
    how="left",
)

subsets = {
    "all_material_sides": set(diag["target_id"]),
    "correct_sides": set(diag.loc[diag["side_correct"], "target_id"]),
    "strong_side_gt_0p2": set(diag.loc[diag["strong_side_gt_0p2"], "target_id"]),
    "very_strong_side_gt_0p5": set(diag.loc[diag["very_strong_side_gt_0p5"], "target_id"]),
}

aggregate_records = []
for subset_name, target_ids in subsets.items():
    subset_rows = rows[rows["target_id"].isin(target_ids)]
    for source, source_frame in subset_rows.groupby("source"):
        aggregate_records.append(
            {
                "subset": subset_name,
                "source": source,
                "rank_bin": "all100",
                **summarize(source_frame).to_dict(),
            }
        )
        if source == "top":
            for limit in (1, 5, 10, 20, 50):
                aggregate_records.append(
                    {
                        "subset": subset_name,
                        "source": source,
                        "rank_bin": f"top{limit}",
                        **summarize(source_frame[source_frame["rank"] <= limit]).to_dict(),
                    }
                )
aggregate = pd.DataFrame.from_records(aggregate_records)
aggregate_path = out_dir / f"completion_side_strong_lexical_aggregate_{step_tag(step)}.csv"
aggregate.to_csv(aggregate_path, index=False)

query_records = []
for target_id, group in rows.groupby("target_id"):
    first = group.iloc[0].to_dict()
    record = {
        key: first.get(key)
        for key in (
            "target_id",
            "base_target_id",
            "completion_side",
            "concept_a",
            "concept_b",
            "active_side_text",
            "margin_1",
            "margin_2",
            "combined_margin",
            "side_margin",
            "side_correct",
            "strong_side_gt_0p2",
            "very_strong_side_gt_0p5",
        )
    }
    for source in ("top", "random"):
        sub = group[group["source"] == source]
        if len(sub):
            summary = summarize(sub)
            for key, value in summary.items():
                record[f"{source}_{key}"] = value
    for limit in (1, 5, 10):
        top = group[(group["source"] == "top") & (group["rank"] <= limit)]
        record[f"top{limit}_any_query_stem_rate"] = float(top["has_query_stem_hit"].mean()) if len(top) else np.nan
        record[f"top{limit}_any_material_keyword_rate"] = (
            float(top["has_material_keyword_hit"].mean()) if len(top) else np.nan
        )
    record["query_stem_enrichment"] = record.get("top_any_query_stem_rate", np.nan) - record.get(
        "random_any_query_stem_rate", np.nan
    )
    record["material_keyword_count_enrichment"] = record.get(
        "top_mean_material_keyword_hit_count", np.nan
    ) - record.get("random_mean_material_keyword_hit_count", np.nan)
    query_records.append(record)

query_summary = pd.DataFrame.from_records(query_records).sort_values(
    ["side_margin", "combined_margin"],
    ascending=[False, False],
)
query_summary_path = out_dir / f"completion_side_strong_lexical_query_summary_{step_tag(step)}.csv"
query_summary.to_csv(query_summary_path, index=False)

for name, mask_col in (
    ("correct_side_items", "side_correct"),
    ("strong_side_gt_0p2_items", "strong_side_gt_0p2"),
    ("very_strong_side_gt_0p5_items", "very_strong_side_gt_0p5"),
):
    diag.loc[diag[mask_col]].sort_values(["side_margin", "combined_margin"], ascending=[False, False]).to_csv(
        out_dir / f"{name}_{step_tag(step)}.csv",
        index=False,
    )

strong_rows = rows[rows["strong_side_gt_0p2"]].copy()
strong_top5 = strong_rows[(strong_rows["source"] == "top") & (strong_rows["rank"] <= 5)].sort_values(
    ["target_id", "rank"]
)
strong_top5_path = out_dir / f"strong_side_gt_0p2_top5_rows_{step_tag(step)}.csv"
strong_top5.to_csv(strong_top5_path, index=False)

supported = strong_top5[strong_top5["has_query_stem_hit"]].sort_values(
    ["query_stem_hit_count", "score"],
    ascending=[False, False],
)
supported_path = out_dir / f"strong_side_gt_0p2_lexically_supported_top5_{step_tag(step)}.csv"
supported.head(100).to_csv(supported_path, index=False)

low = strong_top5[(~strong_top5["has_query_stem_hit"]) & (~strong_top5["has_material_keyword_hit"])].sort_values(
    "score",
    ascending=False,
)
low_path = out_dir / f"strong_side_gt_0p2_high_score_low_lexical_top5_{step_tag(step)}.csv"
low.head(100).to_csv(low_path, index=False)

report = [
    f"# Completion-side material-dynamics lexical audit ({step_tag(step)})",
    "",
    "## Side definitions",
    "",
    f"- all material sides: {len(subsets['all_material_sides'])}",
    f"- correct sides: {len(subsets['correct_sides'])}",
    f"- strong sides, side_margin > 0.2: {len(subsets['strong_side_gt_0p2'])}",
    f"- very strong sides, side_margin > 0.5: {len(subsets['very_strong_side_gt_0p5'])}",
    "",
    "## Aggregate lexical rates",
    "",
    md_table(
        aggregate[aggregate["rank_bin"].isin(["all100", "top1", "top5", "top10"])],
        [
            "subset",
            "source",
            "rank_bin",
            "targets",
            "rows",
            "any_query_stem_rate",
            "any_material_keyword_rate",
            "mean_query_stem_fraction",
            "mean_material_keyword_hit_count",
        ],
        n=80,
    ),
    "",
    "## Strong sides by side margin",
    "",
    md_table(
        query_summary[query_summary["strong_side_gt_0p2"]],
        [
            "target_id",
            "completion_side",
            "concept_a",
            "concept_b",
            "side_margin",
            "combined_margin",
            "top5_any_query_stem_rate",
            "top_any_query_stem_rate",
            "random_any_query_stem_rate",
        ],
        n=25,
    ),
    "",
    "## Strong-side top-5 rows with direct query-stem hits",
    "",
    md_table(
        supported,
        [
            "target_id",
            "rank",
            "score",
            "candidate_id",
            "completion_side",
            "query_stem_hits",
            "material_keyword_hits",
            "text_preview",
        ],
        n=20,
    ),
    "",
    "## Strong-side top-5 rows with no query/material lexical support",
    "",
    md_table(
        low,
        [
            "target_id",
            "rank",
            "score",
            "candidate_id",
            "completion_side",
            "query_stem_hits",
            "material_keyword_hits",
            "text_preview",
        ],
        n=20,
    ),
    "",
]
report_path = out_dir / f"completion_side_strong_lexical_audit_report_{step_tag(step)}.md"
report_path.write_text("\n".join(report), encoding="utf-8")

print(f"rows: {rows_path}")
print(f"aggregate: {aggregate_path}")
print(f"query_summary: {query_summary_path}")
print(f"strong_top5: {strong_top5_path}")
print(f"supported: {supported_path}")
print(f"low_lexical: {low_path}")
print(f"report: {report_path}")
PY

echo "Done."
echo "TrackStar outputs: ${OUT_DIR}"
echo "Lexical audit: ${LEX_DIR}"
