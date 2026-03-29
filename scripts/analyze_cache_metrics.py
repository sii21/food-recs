"""Offline stats from artifacts/data_cache.pkl only (no raw orders.csv)."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def basket_stats(bs: list[list[int]]) -> dict:
    lens = sorted(len(b) for b in bs)
    n = len(lens)
    mean = sum(lens) / n
    p50 = lens[n // 2]
    p90 = lens[int(n * 0.9)]
    flat = (x for b in bs for x in b)
    return {
        "n_baskets": n,
        "mean_size": round(mean, 3),
        "p50": p50,
        "p90": p90,
        "uniq_items": len(set(flat)),
    }


def metrics_one_basket(
    held: int,
    ins: set[int],
    top_order: list[int],
    ks: tuple[int, ...],
) -> tuple[dict[int, int], float]:
    """First filtered rank of held-out; same ordering as TopPopular (freq desc, skip basket)."""
    valid_pos = 0
    for x in top_order:
        if x in ins:
            continue
        if x == held:
            rank = valid_pos + 1
            mrr = 1.0 / rank
            hits = {k: int(rank <= k) for k in ks}
            return hits, mrr
        valid_pos += 1
    return dict.fromkeys(ks, 0), 0.0


def oracle_l1o(
    baskets: list[list[int]],
    top_order: list[int],
    seed: int,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict:
    rng = np.random.default_rng(seed)
    agg_hits = dict.fromkeys(ks, 0)
    mrr_sum = 0.0
    n = 0
    for b in baskets:
        if len(b) < 2:
            continue
        hi = int(rng.integers(len(b)))
        held = b[hi]
        ins = set(b[:hi] + b[hi + 1 :])
        h, m = metrics_one_basket(held, ins, top_order, ks)
        for k in ks:
            agg_hits[k] += h[k]
        mrr_sum += m
        n += 1
    return {
        "n": n,
        "hit": {f"@{k}": agg_hits[k] / n for k in ks},
        "mrr": mrr_sum / n,
    }


def main() -> None:
    cache_path = Path("artifacts/data_cache.pkl")
    if not cache_path.exists():
        raise SystemExit(
            "Missing artifacts/data_cache.pkl — build cache via training pipeline, not 4GB CSV."
        )

    with open(cache_path, "rb") as f:
        c = pickle.load(f)

    tr, te, oot = c["train_baskets"], c["test_baskets"], c["oot_baskets"]
    im = c["item_mapping"]

    st = {x for b in tr for x in b}
    se = {x for b in te for x in b}
    so = {x for b in oot for x in b}

    freq: Counter[int] = Counter()
    for b in tr:
        for x in b:
            freq[x] += 1
    top_order = sorted(freq.keys(), key=lambda x: -freq[x])
    n_items_train = len(freq)
    total_line = sum(freq.values())
    top_shares = {}
    for k in (1, 5, 10, 20, 50, 100, 200):
        acc = sum(freq[it] for it in top_order[:k])
        top_shares[str(k)] = round(acc / total_line, 4)

    rng = np.random.default_rng(42)
    ranks = {it: r + 1 for r, it in enumerate(top_order)}
    sum_r = cnt_r = 0
    for b in te:
        if len(b) < 2:
            continue
        hi = int(rng.integers(len(b)))
        held = b[hi]
        sum_r += ranks.get(held, n_items_train + 1)
        cnt_r += 1
    avg_rank_held = round(sum_r / cnt_r, 2) if cnt_r else None

    n_cat = len(im)
    rand5 = 5 / n_cat if n_cat else 0.0

    out = {
        "source": str(cache_path.resolve()),
        "split_config": "train 365d / test 30d / oot 30d (configs/data/default.yaml)",
        "item_mapping_size": n_cat,
        "unique_items_in_train_baskets": n_items_train,
        "test_items_never_in_train": len(se - st),
        "test_cold_item_frac": round(len(se - st) / len(se), 4),
        "oot_items_never_in_train": len(so - st),
        "oot_cold_item_frac": round(len(so - st) / len(so), 4),
        "train": basket_stats(tr),
        "test": basket_stats(te),
        "oot": basket_stats(oot),
        "train_line_item_mass_topk": top_shares,
        "random_hit5_uniform_over_mapping": round(rand5, 6),
        "avg_popularity_rank_random_heldout_test": avg_rank_held,
        "oracle_filtered_train_freq_l1o_test": oracle_l1o(te, top_order, seed=42),
        "oracle_filtered_train_freq_l1o_oot": oracle_l1o(oot, top_order, seed=42),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
