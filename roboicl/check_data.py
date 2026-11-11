"""Check mixed sources, real RGB decoding, time axes and exact-shape sampling."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import zipfile

import numpy as np

from prepare import camera_count, latent_plan, load_plan, recording_path, required_latents, write_json
from recording_io import read_rgb, validate_timing


def check_record(item):
    episode, record, root = item
    path = recording_path(root, episode, record)
    frames = record["frames"]
    if path.suffix == ".npz":
        with zipfile.ZipFile(path) as archive:
            for name, expected in (("rgb", (frames, 2, 224, 224, 3)), ("state", (frames, 10)), ("action", (frames, 10))):
                with archive.open(name + ".npy") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"Unsupported NPY header: {path}")
                if shape != expected or (name == "rgb" and dtype != np.uint8):
                    raise ValueError(f"Invalid {name} array header: {path}")
        with np.load(path, allow_pickle=False) as data:
            validate_timing(data["timestamp_s"], data["observation_step"], frames, path)
            for name in ("state", "action"):
                values = data[name]
                if values.dtype != np.float32 or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {name} values: {path}")
    else:
        # Decode the first, middle and final head images of every ALOHA source.
        read_rgb(path, frames, [0, frames // 2, frames - 1], 1)
    return episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    plan = load_plan(args.work)
    root = Path(plan["source_root"])
    entries = [(episode, row, root) for episode, row in sorted(plan["records"].items())]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, _ in enumerate(pool.map(check_record, entries), 1):
            if index % 500 == 0:
                print(f"Checked metadata, timing and actions {index}/{len(entries)}", flush=True)
    cases = {}
    for role, episode in required_latents(plan):
        record = plan["records"][episode]
        group = record.get("source_dataset", "legacy"), record.get("kind", "franka"), role
        # Keep a short real recording from every source/embodiment/role.
        if group not in cases or record["frames"] < plan["records"][cases[group]]["frames"]:
            cases[group] = episode
    decoded = []
    for group, episode in sorted(cases.items()):
        record = plan["records"][episode]
        views = camera_count(plan, group[-1])
        ids = [0, record["frames"] // 2, record["frames"] - 1]
        pixels = read_rgb(recording_path(root, episode, record), record["frames"], ids, views)
        decoded.append(dict(group=list(group), episode=episode, shape=list(pixels.shape)))
    from train import SameLengthSampler, length_buckets

    buckets = length_buckets(plan)
    sampler = SameLengthSampler(buckets, 4, 0, 16, 1)
    used = []
    for rank in range(4):
        used.extend(row[0] for batch in SameLengthSampler(buckets, 4, rank, 16, 1) for row in batch)
    if len(used) != len(set(used)) or len(used) != sampler.total_size:
        raise ValueError("Sampler duplicates pairs or disagrees across ranks")
    for episode, record in plan["records"].items():
        if len(latent_plan(record["frames"], "demo")["latent_anchor_indices"]) < 1:
            raise ValueError(f"Empty temporal plan: {episode}")
    retained = Counter(plan["pairs"]["train"][index]["demo_embodiment"] for index in used)
    report = dict(status="passed", checked_records=len(entries), decoded_cases=decoded,
                  train_pairs=len(plan["pairs"]["train"]), evaluation_pairs=len(plan["pairs"]["validation"]),
                  cache_entries=dict(Counter(role for role, _ in required_latents(plan))),
                  batch_per_gpu=16, world_size=4, steps_per_epoch=len(sampler),
                  retained_pairs_per_epoch=sampler.total_size, retained_demo_embodiments=dict(retained),
                  dropped_pairs_per_epoch=len(plan["pairs"]["train"]) - sampler.total_size,
                  cache_validation="source metadata and RGB checked; encoder cache smoke is separate")
    write_json(args.output, report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
