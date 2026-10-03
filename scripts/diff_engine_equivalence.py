"""Compare two engine-equivalence snapshots (scripts/engine_equivalence.ps1) query by query.

    python scripts/diff_engine_equivalence.py <golden_dir> <candidate_dir> [--require results,structure,counters]

Checks, per (file, schema, query):
  results    returned_points and result_checksum          - must hold at EVERY engine step
  structure  structure_digest, num_nodes/leaves, max_depth - holds unless a step changes the tree
  counters   visited_nodes, tested_points, fully_contained - holds unless a step redefines them
Exits 1 if any required check fails, listing the first mismatches so they can be reproduced with
--mode engine-check on the named cloud and schema.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

GROUPS = {
    "results": ("returned_points", "result_checksum"),
    "structure": ("structure_digest", "num_nodes", "num_leaves", "max_depth"),
    "counters": ("visited_nodes", "tested_boxes", "tested_points", "fully_contained_nodes"),
}


def load(path):
    with open(path, newline="") as handle:
        return {(row["schema_name"], row["query_id"]): row for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("golden", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--require", default="results,structure,counters",
                        help="comma list of groups that must match (results,structure,counters)")
    parser.add_argument("--show", type=int, default=8, help="mismatches to print per file and group")
    args = parser.parse_args()

    required = [g.strip() for g in args.require.split(",") if g.strip()]
    unknown = set(required) - set(GROUPS)
    if unknown:
        parser.error(f"unknown group(s): {', '.join(sorted(unknown))}")

    goldens = sorted(args.golden.glob("*_queries.csv"))
    if not goldens:
        sys.exit(f"no *_queries.csv in {args.golden}")

    failed = False
    totals = Counter()
    for golden_path in goldens:
        candidate_path = args.candidate / golden_path.name
        if not candidate_path.exists():
            print(f"{golden_path.name}: MISSING in candidate")
            failed = True
            continue

        golden, candidate = load(golden_path), load(candidate_path)
        missing = golden.keys() - candidate.keys()
        extra = candidate.keys() - golden.keys()
        mismatches = {group: [] for group in GROUPS}
        for key in golden.keys() & candidate.keys():
            g, c = golden[key], candidate[key]
            for group, columns in GROUPS.items():
                # A column only one snapshot has (added by a later engine) is not compared.
                differing = [col for col in columns if col in g and col in c and g[col] != c[col]]
                if differing:
                    mismatches[group].append((key, differing, g, c))

        totals["queries"] += len(golden)
        line = [f"{golden_path.name}: {len(golden)} rows"]
        if missing or extra:
            line.append(f"missing={len(missing)} extra={len(extra)}")
            failed = True
        for group in GROUPS:
            count = len(mismatches[group])
            totals[group] += count
            flag = "" if count == 0 else (" FAIL" if group in required else " (allowed)")
            line.append(f"{group}={count}{flag}")
            if count and group in required:
                failed = True
        print("  ".join(line))

        for group in required:
            for (schema, query), differing, g, c in sorted(mismatches[group])[: args.show]:
                detail = ", ".join(f"{col} {g[col]} -> {c[col]}" for col in differing)
                print(f"    [{group}] {schema} q{query} ({g['query_type']}): {detail}")

    print(f"total: {totals['queries']} rows; " + ", ".join(f"{g}={totals[g]}" for g in GROUPS)
          + f"; required: {','.join(required)} -> {'FAIL' if failed else 'OK'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
