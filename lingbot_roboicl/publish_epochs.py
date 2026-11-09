"""Commit completed LingBot-VA RoboICL adapters from a separate CPU process."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import CommitOperationAdd, HfApi, get_hf_file_metadata, hf_hub_url

ADAPTER = "roboicl_lora.safetensors"
DEFAULT_REPO = "gulucaptain/lingbot-va-roboicl"


def publish(api, repo, revision, run, epoch, step, weights, metadata):
    hasher = hashlib.sha256()
    with weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    checksum = hasher.hexdigest()
    prefix = f"runs/{run.name}"
    destination = f"{prefix}/checkpoints/epoch-{epoch:03d}/{ADAPTER}"
    for attempt in range(1, 6):
        try:
            head = api.repo_info(repo, repo_type="model", revision=revision).sha
            if api.file_exists(repo, destination, repo_type="model", revision=head):
                remote = get_hf_file_metadata(hf_hub_url(repo, destination, revision=head), token=True)
                if remote.etag != checksum:
                    raise ValueError(f"Different weights already exist at {destination}")
                return head

            receipt = json.dumps({"epoch": epoch, "optimizer_step": step, "sha256": checksum})
            operations = [
                CommitOperationAdd(path_in_repo=destination, path_or_fileobj=weights),
                CommitOperationAdd(path_in_repo=f"{prefix}/checkpoints/epoch-{epoch:03d}/epoch.json",
                                   path_or_fileobj=receipt.encode()),
                CommitOperationAdd(path_in_repo=f"{prefix}/roboicl.json",
                                   path_or_fileobj=metadata),
            ]
            commit = api.create_commit(
                repo, operations, repo_type="model", revision=revision, parent_commit=head,
                commit_message=f"RoboICL {run.name} epoch {epoch} (step {step})",
            )
            remote = get_hf_file_metadata(hf_hub_url(repo, destination, revision=commit.oid), token=True)
            if remote.etag != checksum:
                raise ValueError(f"Remote checksum mismatch at {destination}")
            return commit.oid
        except ValueError:
            raise
        except Exception as error:
            if attempt == 5:
                raise
            print(f"Epoch {epoch}: attempt {attempt} failed ({type(error).__name__}); retrying", flush=True)
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--check", action="store_true", help="Check login and repository access")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--wait-seconds", type=int, default=0, help="Maximum wait for each checkpoint")
    args = parser.parse_args()
    if not args.check and (
        args.run is None or not args.epochs or not args.steps_per_epoch
        or min(args.epochs, args.steps_per_epoch) < 1 or args.wait_seconds < 0
    ):
        parser.error("upload requires --run, positive --epochs and --steps-per-epoch")
    api = HfApi(token=True)
    user = api.whoami()["name"]
    repo = api.repo_info(args.repo, repo_type="model", revision=args.revision)
    print(f"Authenticated as {user}; model repo {repo.id}, branch {args.revision}, "
          f"private={repo.private}", flush=True)
    if args.check:
        return

    run = args.run.resolve()
    for epoch in range(1, args.epochs + 1):
        step = epoch * args.steps_per_epoch
        weights = run / "checkpoints" / f"checkpoint_step_{step}" / ADAPTER
        deadline = time.monotonic() + args.wait_seconds
        while not weights.is_file():
            if time.monotonic() >= deadline:
                raise FileNotFoundError(f"Completed adapter not found: {weights}")
            time.sleep(min(10, max(0, deadline - time.monotonic())))
        contract = json.loads((run / "roboicl.json").read_text())
        if contract.get("updates_per_epoch") != args.steps_per_epoch:
            raise ValueError("Uploader epoch length differs from training")
        boundary = json.loads((weights.parent / "restart.json").read_text())
        if (boundary["step"], boundary["epoch"]) != (step, epoch):
            raise ValueError("Checkpoint is outside the requested epoch boundary")
        metadata = json.dumps({key: contract[key] for key in ("plan_sha256", "stats_sha256")}).encode()
        commit = publish(api, repo.id, args.revision, run, epoch, step, weights, metadata)
        print(f"Verified epoch {epoch}, step {step}, commit {commit}", flush=True)


if __name__ == "__main__":
    main()
