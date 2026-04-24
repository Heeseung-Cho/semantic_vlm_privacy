#!/usr/bin/env bash
# Run the 2x2 main ablation on QUERY FULL DATA (1056 images) to produce submission JSONs.
# No eval (query_set has no GT annotations).
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Stage-1 base on query full data (1056 images) ==="
bash "${D}/stage1_base/run.sh"

echo "=== Row 1 : baseline catonly (no cue, no fam, +caption) ==="
bash "${D}/row1_baseline_catonly/run_chain.sh"

echo "=== Row 2 : cue only (+cue, no fam, +caption) ==="
bash "${D}/row2_cue_only/run_chain.sh"

echo "=== Row 3 : FAM only (no cue, +fam, +caption) ==="
bash "${D}/row3_fam_only/run_chain.sh"

echo "=== Row 4 : BEST (+cue, +fam, +caption) ==="
bash "${D}/row4_best_cue_fam/run_chain.sh"

echo "=== done. Submission JSONs (bbox only / bbox+mask):"
for r in row1_baseline_catonly row2_cue_only row3_fam_only row4_best_cue_fam; do
  echo "  ${D}/${r}/stage3/query_submission.json"
  echo "  ${D}/${r}/stage4_sam/query_submission_segm.json"
done
