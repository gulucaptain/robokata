"""Bind the 25k derived pairs to immutable source files and frozen evaluation."""

import hashlib
import json
import re
from pathlib import Path

from prepare import DUAL_ENCODING, SCHEMA, read_json, sha256, write_json


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plan_mixed(root, manifest, work):
    receipt = read_json(manifest / "validation.json")
    for name in ("pairs", "records"):
        if sha256(manifest / f"{name}.jsonl") != receipt[f"{name}_sha256"]:
            raise ValueError(f"Derived {name} manifest hash changed")
    for key, filename in (("new_pairs", "pairs.jsonl"), ("new_records", "records.jsonl"),
                          ("evaluation_pairs", "evaluation_pairs.jsonl")):
        if sha256(root / "manifests" / filename) != receipt["sources_sha256"][key]:
            raise ValueError(f"Source manifest changed: {filename}")
    release = read_json(root / "release_status.json")
    if not release["complete"] or release["version"] != "v2.1-20261110":
        raise ValueError("Expected verified RoboKata v2.1")
    bundle = read_json(root / "bundle.json")
    if sha256(root / "bundle.json") != release["bundle_sha256"]:
        raise ValueError("RoboKata bundle hash changed")
    expected = read_json(root / "DATASET.json")["manifest_sha256"]
    evaluation_path = root / "manifests" / "evaluation_records.jsonl"
    if sha256(evaluation_path) != expected["evaluation_records.jsonl"]:
        raise ValueError("Evaluation recording manifest hash changed")
    source_rows = jsonl(manifest / "records.jsonl")
    records = {}
    for row in source_rows:
        key = row["id"]
        if key in records or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError(f"Unsafe or duplicate recording ID: {key}")
        record = dict(row)
        record["source_sha256"] = row["recording_sha256"]
        record["recording_path"] = str(Path(row["recording_path"]).resolve())
        records[key] = record
    train = jsonl(manifest / "pairs.jsonl")
    if len(train) != 25000 or len({(p["query"], p["demo"]) for p in train}) != 25000:
        raise ValueError("Expected 25,000 unique training pairs")
    if {kind: sum(p["source_dataset"] == kind for p in train) for kind in ("old", "new")} != {"old": 13000, "new": 12000}:
        raise ValueError("Expected 13k old plus 12k new pairs")
    # Build a separate namespace for frozen evaluation and keep its source roles.
    for row in jsonl(evaluation_path):
        key = "eval__" + row["id"]
        filename = row["files"]["episode.hdf5" if row["kind"] == "aloha" else "policy_recording.npz"]
        records[key] = dict(row, id=key, source_dataset="evaluation",
                            recording_path=str(root / filename), source_sha256=bundle["files"][filename]["sha256"])
    validation = []
    for pair in jsonl(root / "manifests" / "evaluation_pairs.jsonl"):
        validation.append({**pair, "query": "eval__" + pair["query"], "demo": "eval__" + pair["demo"]})
    if len(validation) != 1200:
        raise ValueError("Expected 1,200 frozen evaluation pairs")
    families = {}
    for split, pairs in (("train", train), ("validation", validation)):
        used = set()
        seen = set()
        for pair in pairs:
            query, demo = (records[pair[role]] for role in ("query", "demo"))
            key = (pair["query"], pair["demo"])
            if key in seen or pair["query"] == pair["demo"] or pair.get("self_pair") is not False:
                raise ValueError("Duplicate or self pair")
            seen.add(key)
            if query["task_id"] != demo["task_id"] or query["task_id"] != pair["task_id"]:
                raise ValueError("Pair task mismatch")
            if query["kind"] != "franka" or query.get("role", "query") != "query":
                raise ValueError("Only Franka queries are supervised")
            for record, role in ((query, "query"), (demo, "demo")):
                if "role" in record and record["role"] != role:
                    raise ValueError("Frozen evaluation role mismatch")
                if "roles" in record and role not in record["roles"]:
                    raise ValueError("Recording role is not permitted")
                if record["split"] != ("train" if split == "train" else "evaluation"):
                    raise ValueError("Pair crosses the declared split")
                if not Path(record["recording_path"]).is_file():
                    raise FileNotFoundError(record["recording_path"])
                used.update(record[k] for k in ("family", "lineage_family") if record.get(k))
            if query["family"] == demo["family"] or query.get("lineage_family", query["family"]) == demo.get("lineage_family", demo["family"]):
                raise ValueError("Demo and query share a physical family")
        families[split] = used
    if families["train"] & families["validation"]:
        raise ValueError("Training/evaluation physical family overlap")
    identity = {"derived_pairs": sha256(manifest / "pairs.jsonl"), "derived_records": sha256(manifest / "records.jsonl"),
                "evaluation_pairs": sha256(root / "manifests/evaluation_pairs.jsonl"),
                "evaluation_records": sha256(evaluation_path), "release_bundle": release["bundle_sha256"]}
    plan = dict(schema=SCHEMA, encoding=DUAL_ENCODING, demo_top_view=True,
                dataset="robokata_mixed25000_v1", source_root=str(root), source_manifests=identity,
                bundle_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                pairs={"train": train, "validation": validation}, records=records,
                validation_families=sorted(families["validation"]), split_seed=receipt["seed"],
                evaluation_protocol="frozen_robokata_v2.1_1200_pairs")
    destination = work / "plan.json"
    if destination.exists() and read_json(destination) != plan:
        raise ValueError("Work directory already contains another plan")
    write_json(destination, plan)
    print(json.dumps({"plan": str(destination), "train_pairs": len(train), "evaluation_pairs": len(validation),
                      "records": len(records), "demo_views": 1, "query_views": 2}, indent=2))
