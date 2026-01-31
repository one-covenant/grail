#!/usr/bin/env python3
"""
Deduplicate and combine sparsity CSV files.

Strategy:
1. Load all CSV files with source tracking
2. For each experiment config (model, lr, iter), determine completeness
3. For duplicates, prefer data from the most complete source
4. Output a single clean CSV

Completeness criteria:
- Full steps (399 steps = complete, <399 = incomplete)
- All seeds (4 seeds = complete)
- All k values (6 k values = complete)
"""

import pandas as pd
from pathlib import Path
import argparse


def load_csvs_with_source(csv_paths: list[str]) -> pd.DataFrame:
    """Load all CSVs and tag with source file."""
    dfs = []
    for path in csv_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            df['_source'] = path
            dfs.append(df)
            print(f"Loaded {len(df):,} rows from {path}")
        else:
            print(f"WARNING: {path} not found")

    if not dfs:
        raise ValueError("No CSV files found")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows before dedup: {len(combined):,}")
    return combined


def analyze_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze completeness of each experiment configuration per source.
    Returns DataFrame with completeness scores.
    """
    # Group by experiment config + source
    group_cols = ['model_family', 'model_size', 'learning_rate', 'iteration_num', '_source']

    completeness = []
    for keys, group in df.groupby(group_cols):
        model_family, model_size, lr, iter_num, source = keys

        seeds = group['seed'].nunique()
        k_values = group['k'].nunique()
        steps = group[group['k'] == 1]['step'].nunique() if 1 in group['k'].values else 0
        max_step = group['step'].max()

        # Completeness score: higher is better
        # Full experiment: 4 seeds × 6 k × 399 steps = 9576 rows
        score = seeds * k_values * steps

        completeness.append({
            'model_family': model_family,
            'model_size': model_size,
            'learning_rate': lr,
            'iteration_num': iter_num,
            '_source': source,
            'seeds': seeds,
            'k_values': k_values,
            'steps': steps,
            'max_step': max_step,
            'completeness_score': score,
            'is_complete': (seeds >= 4 and k_values >= 6 and steps >= 399)
        })

    return pd.DataFrame(completeness)


def deduplicate(df: pd.DataFrame, completeness_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by keeping data from the most complete source for each experiment.
    For experiments that exist in multiple sources with different seeds, combine them.
    """
    # Create experiment key columns
    exp_cols = ['model_family', 'model_size', 'learning_rate', 'iteration_num']
    row_key_cols = exp_cols + ['seed', 'k', 'step']

    # For each experiment, rank sources by completeness
    source_priority = completeness_df.sort_values(
        ['model_family', 'model_size', 'learning_rate', 'iteration_num', 'completeness_score'],
        ascending=[True, True, True, True, False]
    )

    # Create a mapping: (experiment) -> ranked list of sources
    exp_to_sources = {}
    for _, row in source_priority.iterrows():
        key = (row['model_family'], row['model_size'], row['learning_rate'], row['iteration_num'])
        if key not in exp_to_sources:
            exp_to_sources[key] = []
        exp_to_sources[key].append({
            'source': row['_source'],
            'score': row['completeness_score'],
            'seeds': row['seeds'],
            'steps': row['steps']
        })

    print("\n=== SOURCE PRIORITY BY EXPERIMENT ===")
    for exp, sources in sorted(exp_to_sources.items()):
        print(f"\n{exp[0]} {exp[1]} lr={exp[2]:.0e} iter={exp[3]}:")
        for i, s in enumerate(sources):
            priority = "PRIMARY" if i == 0 else "SUPPLEMENT"
            print(f"  [{priority}] {Path(s['source']).name}: score={s['score']}, seeds={s['seeds']}, steps={s['steps']}")

    # Deduplicate: for each unique row key, keep from highest priority source
    # But also include supplementary seeds that don't exist in primary source

    # Add experiment key to df
    df['_exp_key'] = df.apply(
        lambda r: (r['model_family'], r['model_size'], r['learning_rate'], r['iteration_num']),
        axis=1
    )

    # Add row key
    df['_row_key'] = df.apply(
        lambda r: (r['model_family'], r['model_size'], r['learning_rate'], r['iteration_num'],
                   r['seed'], r['k'], r['step']),
        axis=1
    )

    # Assign source priority (lower = better)
    def get_source_priority(row):
        exp_key = row['_exp_key']
        source = row['_source']
        if exp_key in exp_to_sources:
            for i, s in enumerate(exp_to_sources[exp_key]):
                if s['source'] == source:
                    return i
        return 999

    df['_source_priority'] = df.apply(get_source_priority, axis=1)

    # Sort by row key and source priority, keep first
    df_sorted = df.sort_values(['_row_key', '_source_priority'])
    df_dedup = df_sorted.drop_duplicates(subset=row_key_cols, keep='first')

    # Clean up temp columns
    df_dedup = df_dedup.drop(columns=['_source', '_exp_key', '_row_key', '_source_priority'])

    return df_dedup


def validate_output(df: pd.DataFrame):
    """Validate the deduplicated output."""
    print("\n=== VALIDATION ===")

    # Check for any remaining duplicates
    row_key_cols = ['model_family', 'model_size', 'learning_rate', 'iteration_num', 'seed', 'k', 'step']
    dupes = df.duplicated(subset=row_key_cols, keep=False)
    if dupes.any():
        print(f"WARNING: {dupes.sum()} duplicate rows remain!")
    else:
        print("OK: No duplicates")

    # Summary by experiment
    print("\n=== FINAL DATA SUMMARY ===")
    for (fam, size, lr, iter_num), g in df.groupby(['model_family', 'model_size', 'learning_rate', 'iteration_num']):
        seeds = sorted(g['seed'].unique())
        k_vals = sorted(g['k'].unique())
        steps = g[g['k'] == 1]['step'].nunique()
        print(f"{fam} {size} lr={lr:.0e} iter={iter_num}: {len(seeds)} seeds, {len(k_vals)} k, {steps} steps")


def main():
    parser = argparse.ArgumentParser(description='Deduplicate and combine sparsity CSV files')
    parser.add_argument('--input', '-i', nargs='+', default=[
        'data/sparsity_k_step.csv',
        'data/sparsity_k_step_remaining.csv',
        'data/sparsity_k_step_new_experiments.csv',
    ], help='Input CSV files')
    parser.add_argument('--output', '-o', default='data/sparsity_k_step_combined.csv',
                        help='Output CSV file')
    parser.add_argument('--dry-run', action='store_true', help='Analyze only, do not write')
    args = parser.parse_args()

    # Load all CSVs
    df = load_csvs_with_source(args.input)

    # Analyze completeness
    completeness = analyze_completeness(df)
    print("\n=== COMPLETENESS ANALYSIS ===")
    print(completeness.to_string())

    # Deduplicate
    df_clean = deduplicate(df, completeness)
    print(f"\nRows after dedup: {len(df_clean):,}")

    # Validate
    validate_output(df_clean)

    # Save
    if not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
    else:
        print("\nDry run - no file written")


if __name__ == '__main__':
    main()
