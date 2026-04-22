"""Post-process a Stage-1 output so that semantic_categories and proposal_prompts
are replaced by the FULL family categories of the declared route_type.

This lets Stage-2 G-DINO and Stage-3 VLM see all siblings of the declared route as the
candidate space (rather than only the VLM's top-1 pick).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from semantic.family_config import get_family_categories, set_active_family_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_path', required=True, help='Stage-1 json output to expand')
    ap.add_argument('--output_path', required=True)
    ap.add_argument('--family_config', required=True)
    args = ap.parse_args()

    set_active_family_config(args.family_config)

    data = json.loads(Path(args.input_path).read_text())
    recs = data if isinstance(data, list) else data.get('records', [])

    n_changed = 0
    for r in recs:
        if r.get('null_likely'):
            continue
        route = r.get('route_type', '')
        fam_cats = get_family_categories(route) or []
        if not fam_cats:
            continue
        # Preserve VLM-pick ordering, then append siblings (so VLM's confidence is rank-1)
        base_cats = r.get('semantic_categories', []) or []
        ordered = [c for c in base_cats if c in fam_cats]
        ordered += [c for c in fam_cats if c not in ordered]
        r['semantic_categories'] = ordered
        # Rebuild proposal_prompts: keep original cue parts that aren't category names,
        # plus all family category names.
        old_prompts = r.get('proposal_prompts', []) or []
        non_cat_prompts = [p for p in old_prompts if p not in fam_cats and p not in base_cats]
        r['proposal_prompts'] = ordered + non_cat_prompts
        n_changed += 1

    out = data if isinstance(data, list) else data
    if isinstance(out, list):
        out = recs
    else:
        out['records'] = recs
    Path(args.output_path).write_text(json.dumps(out, indent=2))
    print(f'Expanded {n_changed} records. Saved: {args.output_path}')


if __name__ == '__main__':
    main()
