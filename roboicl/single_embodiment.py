"""Contract for the original Franka-only RoboICL reading ablation."""

import hashlib
import json


def audit_plan(plan):
    """Reject mixed25k and altered splits; do not filter pairs at train time."""
    records = plan["records"]
    pairs = plan["pairs"]
    if len(records) != 3000 or {k: len(v) for k, v in pairs.items()} != {"train": 21860, "validation": 260}:
        raise ValueError("A_latent requires the original Franka-only plan: 3000 recordings, 21860 train / 260 validation pairs")
    if plan.get("self_pairs_excluded") != 3000 or plan.get("cross_split_excluded") != 4880:
        raise ValueError("Unexpected original RoboICL pair exclusions")
    if plan.get("split_seed") != 20261105 or len(plan.get("validation_families", [])) != 50:
        raise ValueError("A_latent requires the original seed-20261105 held-out-family split")
    for episode, record in records.items():
        # The legacy release has no kind field. Its immutable recording IDs
        # identify Franka; explicit embodiment metadata must agree when present.
        if not episode.startswith("FR2P_") or record.get("kind", "franka") != "franka":
            raise ValueError(f"Non-Franka recording in A_latent data: {episode}")
    split_families = {}
    for split, rows in pairs.items():
        seen = set()
        families = set()
        for row in rows:
            demo, query = row["demo"], row["query"]
            if demo == query or (demo, query) in seen:
                raise ValueError("Duplicate or self pair in A_latent data")
            seen.add((demo, query))
            for episode in (demo, query):
                record = records[episode]
                if record["task_id"] != row["task_id"]:
                    raise ValueError("A_latent pair task mismatch")
                families.add(record["family"])
        split_families[split] = families
    if split_families["train"] & split_families["validation"]:
        raise ValueError("A_latent training/validation families overlap")
    held_out = set(plan["validation_families"])
    if len(split_families["train"]) != 450 or split_families["train"] & held_out:
        raise ValueError("Expected 450 training families outside the declared holdout")
    # Three of the 50 held-out families have no retained non-self pair in
    # the original plan. Pair coverage need not equal the declared holdout.
    if not split_families["validation"] <= held_out:
        raise ValueError("Validation pairs escape the declared holdout")
    return {
        "schema": "roboicl_franka_only_21860_v1",
        "recordings": len(records),
        "train_pairs": len(pairs["train"]),
        "validation_pairs": len(pairs["validation"]),
        "train_query_recordings": len({p["query"] for p in pairs["train"]}),
        "train_demo_recordings": len({p["demo"] for p in pairs["train"]}),
        "canonical_plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
