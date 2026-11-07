"""Train Zero-WAM on verified RoboICL pairs with full-episode batches."""

import argparse
from collections import defaultdict
import math
import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import torch

from examples.baselines.zero_wam.zero_wam import cache, training
from examples.baselines.zero_wam.zero_wam.action_space import EpisodeActionSpace
from examples.baselines.zero_wam.zero_wam.config import ZeroWAMConfig

from pilot import RoboICLDataset
from prepare import (
    ACTION_CHANNELS,
    ACTION_REPRESENTATION,
    latent_path,
    load_plan,
    read_json,
    required_latents,
    sha256,
)


GLOBAL_BATCH = 16


class FullRoboICLDataset(RoboICLDataset):
    """Bind each query to its published demonstration using a singleton candidate."""

    def __init__(self, manifest_path, empty_text_path, seed, action_per_frame,
                 normalization_path, sampling_mode, action_history_mode, **_):
        self.manifest_path = Path(manifest_path)
        self.manifest = plan = load_plan(self.manifest_path.parent)
        work = self.manifest_path.parent
        pairs = plan["pairs"]["train"]
        demo_ids = sorted({pair["demo"] for pair in pairs})
        self.records = [
            {"repo_id": pair["query"], "human_repo_id": pair["demo"],
             "cache_path": str(latent_path(work, "query", pair["query"]))}
            for pair in pairs
        ]
        self.humans = [{"repo_id": demo, "cache_path": str(latent_path(work, "demo", demo))}
                       for demo in demo_ids]
        self.repo_indices = {record: index for index, record in enumerate(sorted({p["query"] for p in pairs}))}
        self.humans_by_repo = {demo: [index] for index, demo in enumerate(demo_ids)}
        self.action_channels = list(ACTION_CHANNELS)
        self.action_space = EpisodeActionSpace(read_json(Path(normalization_path)))
        self.action_per_frame = action_per_frame
        self.sampling_mode = sampling_mode
        self.action_history_mode = action_history_mode
        self.seed = seed
        self.payloads = OrderedDict()
        self.empty_text = torch.load(empty_text_path, map_location="cpu", weights_only=True)


class Batch16Sampler:
    """Shard each epoch's shuffled directed pairs without padding or partial updates."""

    def __init__(self, length, world, rank, seed=1):
        if not 0 <= rank < world or GLOBAL_BATCH % world:
            raise ValueError("Invalid rank or world size for effective batch 16")
        self.world, self.rank, self.seed = world, rank, seed
        self.epoch, self.start_batch = 0, 0
        self.length = length
        self.total_size = length // GLOBAL_BATCH * GLOBAL_BATCH
        if self.total_size == 0:
            raise ValueError("Need at least 16 pairs and a rank count that divides 16")

    def __len__(self):
        return self.total_size // self.world

    def set_epoch(self, epoch):
        self.epoch, self.start_batch = epoch, 0

    def sample_offset_after(self, batch_index):
        return (batch_index + 1) * self.world

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.length, generator=generator).tolist()
        indices = order[self.rank:self.total_size:self.world]
        for batch_index in range(self.start_batch, len(indices)):
            yield [(indices[batch_index], 1.0, self.epoch, batch_index, GLOBAL_BATCH, GLOBAL_BATCH // self.world)]


def length_buckets(plan):
    buckets = defaultdict(list)
    for index, pair in enumerate(plan["pairs"]["train"]):
        frames = [plan["records"][pair[role]]["frames"] for role in ("query", "demo")]
        query = 1 + math.ceil(frames[0] / 16)
        demo = 1 + math.ceil((max(1, round(frames[1] * 12 / 50)) - 1) / 4)
        buckets[query, demo].append(index)
    return dict(buckets)


class SameLengthSampler:
    """Shuffle exact-shape batches to avoid exposing padding to the model."""

    def __init__(self, buckets, world, rank, batch, seed):
        self.buckets, self.world, self.rank = buckets, world, rank
        self.batch, self.seed = batch, seed
        self.epoch, self.start_batch = 0, 0
        self.steps = sum(len(rows) // batch for rows in buckets.values()) // world
        self.total_size = self.steps * batch * world

    def __len__(self):
        return self.steps

    def set_epoch(self, epoch):
        self.epoch, self.start_batch = epoch, 0

    def sample_offset_after(self, batch_index):
        return (batch_index + 1) * self.batch * self.world

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        batches = []
        for rows in self.buckets.values():
            shuffled = [rows[i] for i in torch.randperm(len(rows), generator=rng).tolist()]
            batches.extend(shuffled[i:i + self.batch] for i in range(0, len(rows) - self.batch + 1, self.batch))
        order = torch.randperm(len(batches), generator=rng).tolist()
        for step in range(self.start_batch, self.steps):
            rows = batches[order[step * self.world + self.rank]]
            yield [(index, 1.0, self.epoch, step, self.batch * self.world, 1) for index in rows]


class RoboICLConfig(ZeroWAMConfig):
    def selections(self):
        # The verified cache/plan replaces the IG task registry.
        return [], []

    def code_resources(self):
        # RoboICL reads its own verified plan. Keep the provenance required
        # for checkpoint compatibility without copying task descriptions.
        return read_json(Path(__file__).with_name("backend_source_resources.json"))


def verified_plan(work, model):
    plan_path = work / "plan.json"
    stats_path = work / "stats.json"
    plan = load_plan(work)
    receipt = read_json(work / "verified-cache.json")
    if (receipt["plan_sha256"], receipt["stats_sha256"], receipt["model_path"]) != (
        sha256(plan_path), sha256(stats_path), str(model)
    ):
        raise ValueError("Preparation plan, statistics, or model differ from the verified cache")
    stats = read_json(stats_path)
    if stats["representation"] != ACTION_REPRESENTATION or stats["plan_sha256"] != receipt["plan_sha256"]:
        raise ValueError("Expected the verified absolute RoboICL train-only normalization")
    if sha256(model / "vae/config.json") != receipt["contract"]["encoder"]["config.json"]:
        raise ValueError("VAE config differs from the verified cache")
    expected = {latent_path(work, role, episode).relative_to(work).as_posix()
                for role, episode in required_latents(plan)}
    if receipt["latents"].keys() != expected:
        raise ValueError("Verified cache receipt does not cover every selected role/recording")
    for path, stamp in receipt["latents"].items():
        info = (work / path).stat()
        if [info.st_size, info.st_mtime_ns] != stamp:
            raise ValueError(f"Cache changed since verification: {path}")
    return plan, receipt


def resume_progress(checkpoint, config, samples_per_epoch):
    sampler = Batch16Sampler(samples_per_epoch, config["world_size"], config["rank"])
    batches = len(sampler)
    accumulation = config["gradient_accumulation_steps"]
    epoch, next_batch = (checkpoint["progress"][key] for key in ("epoch", "next_batch"))
    if epoch < 0 or not 0 <= next_batch < batches or next_batch % accumulation:
        raise ValueError("Checkpoint cursor is not at a RoboICL optimizer boundary")
    step = epoch * (batches // accumulation) + next_batch // accumulation
    if step != checkpoint["global_step"] or step > config["num_steps"]:
        raise ValueError("Checkpoint step disagrees with the sampler cursor or requested update budget")
    progress = {"epoch": epoch, "next_batch": next_batch, "sample_offset": next_batch * sampler.world}
    return progress, config["num_steps"]


def install_bridge(work, plan, receipt, max_steps, batch_per_gpu=1, *,
                   demo_cache=None, demo_dropout=0.3, initialize_from_a=False,
                   semantic_cache=None, semantic_weight=0.5):
    native_config = training.native_config
    prepare_episode = training.prepare_episode
    atomic_json = training.atomic_json
    setup_data = training.setup_data
    adapt_transformer = training.adapt_transformer
    buckets = length_buckets(plan) if batch_per_gpu > 1 else None
    if semantic_cache is not None:
        from semantic_features import DIM, GRID, SCHEMA, load_query_features
        from examples.baselines.zero_wam.zero_wam.wan_va.semantic_ifp import (
            SemanticIFPHead, future_anchor_pairs,
        )

        semantic_digest = sha256(semantic_cache)
        query_rows, targets, target_valid, semantic_encoder = load_query_features(semantic_cache, work, plan)
        pair_rows = torch.tensor([query_rows[pair["query"]] for pair in plan["pairs"]["train"]])
    if demo_cache is not None:
        from demo_attention import SPEC, add_to_adapter_report, attach_demo_attention
        from demo_features import load_features

        cache_digest = sha256(demo_cache)
        feature_ids, features = load_features(demo_cache, work, plan)
        if semantic_cache is not None:
            demo_encoder = torch.load(demo_cache, map_location="cpu", weights_only=True, mmap=True)
            if demo_encoder["encoder_sha256"] != semantic_encoder:
                raise ValueError("A and B must use the same DINOv2 checkpoint")
        train_demos = sorted({pair["demo"] for pair in plan["pairs"]["train"]})
        demo_rows = torch.tensor([feature_ids[episode] for episode in train_demos])

        def adapt_with_demo(model, config):
            init_path = config.ig_initialization if initialize_from_a else None
            report = adapt_transformer(
                model, {**config, "ig_initialization": None} if init_path else config
            )
            attach_demo_attention(model, SPEC)
            add_to_adapter_report(model, report)
            if init_path:
                checkpoint = torch.load(init_path, map_location="cpu", weights_only=False, mmap=True)
                weights = checkpoint["adapter_state_dict"]
                if ({name: list(tensor.shape) for name, tensor in weights.items()}
                        != report["trainable_shapes"]):
                    raise ValueError("Version A warm-start weights do not match the trainable parameters")
                model.load_state_dict(weights, strict=False)
                report["initialized_from"] = init_path
            if semantic_cache is not None:
                retired = ("mcp_blocks.", "mcp_projections.")
                report["lora_parameters"] -= sum(
                    p.numel() for name, p in model.named_parameters()
                    if p.requires_grad and name.startswith(retired)
                )
                report["lora_targets"] = [
                    name for name in report["lora_targets"] if not name.startswith(retired)
                ]
                del model.mcp_blocks
                del model.mcp_projections
                model.semantic_ifp = SemanticIFPHead(model.inner_dim, config.num_mcp_modules, DIM, GRID)
                report["trainable_shapes"] = {
                    name: list(p.shape) for name, p in model.named_parameters() if p.requires_grad
                }
                report["semantic_ifp_parameters"] = sum(p.numel() for p in model.semantic_ifp.parameters())
                report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return report

        training.adapt_transformer = adapt_with_demo

    def make_sampler(length, world, rank, seed=1):
        if buckets is not None:
            return SameLengthSampler(buckets, world, rank, batch_per_gpu, seed)
        return Batch16Sampler(length, world, rank, seed)

    def setup_batched_data(trainer):
        trainer.config.batch_size = batch_per_gpu
        setup_data(trainer)

    def resume_batched_progress(checkpoint, config, _):
        sampler = make_sampler(train_count, config["world_size"], config["rank"])
        epoch, next_batch = (checkpoint["progress"][key] for key in ("epoch", "next_batch"))
        step = epoch * len(sampler) + next_batch
        if not 0 <= next_batch < len(sampler) or step != checkpoint["global_step"] or step > config["num_steps"]:
            raise ValueError("Checkpoint cursor disagrees with the same-length sampler")
        progress = dict(epoch=epoch, next_batch=next_batch, sample_offset=next_batch * config["world_size"] * batch_per_gpu)
        return progress, config["num_steps"]
    train_count = len(plan["pairs"]["train"])
    identity = hashlib.sha256((receipt["plan_sha256"] + receipt["stats_sha256"]).encode()).hexdigest()

    def build_config(config, _manifest):
        manifest = {"sim_episodes": plan["pairs"]["train"], "records_sha256": identity}
        settings = native_config(config, manifest)
        world = settings.world_size
        steps = len(make_sampler(train_count, world, settings.rank)) if buckets is not None else train_count // GLOBAL_BATCH
        if steps < 1:
            raise ValueError("Insufficient full batches for the selected topology")
        total = steps * config.total_epochs
        settings.gradient_accumulation_steps = 1 if buckets is not None else GLOBAL_BATCH // world
        settings.ig_effective_batch = world * batch_per_gpu if buckets is not None else GLOBAL_BATCH
        settings.ig_lr_decay_start_step = config.warmup_steps
        settings.ig_lr_decay_steps = max(1, total - config.warmup_steps)
        settings.num_steps = total
        if max_steps is not None:
            if max_steps > total:
                raise ValueError("--max-steps exceeds the full training schedule")
            settings.num_steps = max_steps
            settings.ig_max_steps = max_steps
        settings.drop_icl = 0.0
        settings.ig_action_representation = ACTION_REPRESENTATION
        settings.ig_app_config["action_representation"] = ACTION_REPRESENTATION
        settings.ig_roboicl_channels = list(ACTION_CHANNELS)
        settings.ig_roboicl_cache_sha256 = sha256(work / "verified-cache.json")
        if demo_cache is not None:
            settings.ig_demo_attention = SPEC
            settings.ig_demo_cache_sha256 = cache_digest
            settings.ig_demo_dropout = demo_dropout
        if semantic_cache is not None:
            settings.ig_semantic_ifp = {"schema": SCHEMA, "weight": semantic_weight}
            settings.ig_semantic_ifp_cache_sha256 = semantic_digest
        return settings

    def prepare_with_accumulation(trainer, batch, original):
        inputs = prepare_episode(trainer, batch, original)
        if demo_cache is not None:
            rows = demo_rows[batch["ig_human_index"].cpu()]
            demo = features[rows].to(device=trainer.device, dtype=trainer.dtype)
            keep = torch.rand((len(demo), 1, 1), device=demo.device) >= demo_dropout
            # A dropped demo has zero K/V under the bias-free projections.
            inputs["demo_features"] = demo * keep
        if semantic_cache is not None:
            rows = pair_rows[batch["ig_sample_index"].cpu()]
            frames = batch["latents"].shape[2]
            valid = target_valid[rows, :frames]
            inputs["semantic_ifp_features"] = targets[rows, :frames].to(trainer.device)
            inputs["semantic_ifp_valid"] = valid.to(trainer.device)
            trainer.ig_latest_shape["semantic_ifp_valid_frames"] = [
                mask.sum(1).tolist()
                for _, mask in future_anchor_pairs(
                    valid, trainer.config.num_mcp_modules,
                    inputs["chunk_size"], trainer.config.future_chunk_stride,
                )
            ]
        trainer.ig_group_count = trainer.config.ig_effective_batch
        return inputs

    def save_with_contract(path, value):
        if path.name == "sampling_contract.json":
            value.update(
                epoch="one shuffled presentation per retained official directed pair",
                updates_per_epoch=(len(make_sampler(train_count, int(os.environ.get("WORLD_SIZE", "1")), 0))
                                   if buckets is not None else train_count // GLOBAL_BATCH),
                robot_episodes=len({pair["query"] for pair in plan["pairs"]["train"]}),
                human_episodes=len({pair["demo"] for pair in plan["pairs"]["train"]}),
                human_pairing="exact published demo -> query, non-self and same split",
                action_representation=ACTION_REPRESENTATION,
                action_channel_offset=None,
                action_channels=list(ACTION_CHANNELS),
                padding=("none; discard incomplete exact-shape batches" if buckets is not None
                         else "none; drop epoch tail of fewer than 16 pairs"),
                normalization="unique training query recordings only",
            )
            if semantic_cache is not None:
                value["targets"] = "video/action flow-matching and future query DINO cosine loss"
                value["mcp_source"] = "real query VAE anchors from the verified training split"
        return atomic_json(path, value)

    cache.require_complete = lambda config: plan
    training.native_config = build_config
    training.IGDataset = FullRoboICLDataset
    training.EpisodeBatchSampler = make_sampler
    training.setup_data = setup_batched_data
    training.prepare_episode = prepare_with_accumulation
    training.resume_sample_progress = resume_batched_progress if buckets is not None else resume_progress
    training.atomic_json = save_with_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--empty-text", required=True, type=Path)
    parser.add_argument("--batch-per-gpu", type=int, default=1)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--warmup-steps", default=10, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--initialize-from", type=Path, help="Warm-start from a baseline or A adapter checkpoint")
    parser.add_argument("--demo-features", type=Path, help="Enable A using the precomputed DINOv2 cache")
    parser.add_argument("--demo-dropout", type=float, default=0.3)
    parser.add_argument("--semantic-features", type=Path, help="Replace latent MCP with B1 query DINO targets")
    parser.add_argument("--semantic-weight", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.batch_per_gpu < 1:
        parser.error("--batch-per-gpu must be positive")
    if args.batch_per_gpu == 1 and (not 0 < world <= GLOBAL_BATCH or GLOBAL_BATCH % world):
        parser.error("WORLD_SIZE must divide 16 for one trajectory per rank per microstep")
    if args.epochs < 1 or args.warmup_steps < 0 or args.learning_rate <= 0:
        parser.error("epochs and learning rate must be positive, warmup nonnegative")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if not 0 <= args.demo_dropout < 1:
        parser.error("--demo-dropout must be in [0, 1)")
    if args.semantic_features and not args.demo_features:
        parser.error("--semantic-features requires --demo-features for A+B")
    if not math.isfinite(args.semantic_weight) or args.semantic_weight <= 0:
        parser.error("--semantic-weight must be finite and positive")
    os.environ.setdefault("WORLD_SIZE", str(world))
    work, model = args.work.resolve(), args.model.resolve()
    plan, receipt = verified_plan(work, model)
    if len(plan["pairs"]["train"]) < GLOBAL_BATCH:
        parser.error("need at least 16 training pairs")
    demo_cache = args.demo_features.resolve() if args.demo_features else None
    semantic_cache = args.semantic_features.resolve() if args.semantic_features else None
    initialize_from_a = False
    checkpoint_path = args.resume_from or args.initialize_from
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        previous = checkpoint["native_config"]
        if (previous.get("ig_roboicl_cache_sha256"), previous.get("model_path")) != (
            sha256(work / "verified-cache.json"), str(model)
        ):
            parser.error("Checkpoint uses a different RoboICL cache or base model")
        checkpoint_has_demo = "ig_demo_attention" in previous
        checkpoint_has_semantic = "ig_semantic_ifp" in previous
        if args.resume_from and (
            checkpoint_has_demo != (demo_cache is not None)
            or checkpoint_has_semantic != (semantic_cache is not None)
        ):
            parser.error("Resume with the same Robot-ICL architecture as the checkpoint")
        if args.initialize_from and checkpoint_has_semantic:
            parser.error("Use --resume-from for an A+B checkpoint")
        if args.initialize_from and checkpoint_has_demo:
            from demo_attention import SPEC

            if (demo_cache is None or previous["ig_demo_attention"] != SPEC
                    or previous["ig_demo_cache_sha256"] != sha256(demo_cache)):
                parser.error("A warm-start requires the same demo architecture and feature cache")
            initialize_from_a = True
        del checkpoint
    install_bridge(work, plan, receipt, args.max_steps, args.batch_per_gpu,
                   demo_cache=demo_cache, demo_dropout=args.demo_dropout,
                   initialize_from_a=initialize_from_a,
                   semantic_cache=semantic_cache, semantic_weight=args.semantic_weight)
    training.run(RoboICLConfig(
        model_path=str(model),
        empty_text_embedding=str(args.empty_text.resolve()),
        human_root=plan["source_root"], sim_root=plan["source_root"],
        cache_dir=str(work), manifest=str(work / "plan.json"),
        run_dir=str(args.run_dir.resolve()), normalization_path=str(work / "stats.json"),
        batch_size=1, effective_batch_size=world,
        total_epochs=args.epochs, warmup_steps=args.warmup_steps, save_epoch_freq=1,
        learning_rate=args.learning_rate, seed=args.seed, num_workers=args.num_workers,
        activation_checkpointing=True,
        resume_from=str(args.resume_from.resolve()) if args.resume_from else None,
        initialize_from=str(args.initialize_from.resolve()) if args.initialize_from else None,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
