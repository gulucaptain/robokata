"""Train and evaluate LingBot-VA using RoboICL demonstrations as context."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from lora import ADAPTER_FILE
from roboicl_data import ACTION_CHANNELS, ACTION_INTERVAL, RoboICLDataset


def main():
    root = Path(__file__).resolve().parent.parent
    sys.path[:0] = [str(root / "src"), str(root / "src/wan_va")]
    os.chdir(root / "src")

    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inspect", "train", "infer"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--lora-rank", type=int, default=8, help="Use 0 for full fine-tuning")
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--demo-index", type=int, help="Replace the demo with that held-out pair's demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in (args.work, args.output, args.checkpoint):
        if path and not path.is_absolute():
            parser.error("use absolute paths")
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    if min(args.steps, args.save_interval, args.accumulation) < 1 or min(args.workers, args.warmup_steps) < 0:
        parser.error("invalid training settings")
    if not 0 <= args.lora_rank <= 3072 or not 0 < args.lora_alpha < float("inf"):
        parser.error("invalid LoRA settings")
    if args.learning_rate is not None and not 0 < args.learning_rate < float("inf"):
        parser.error("invalid learning rate")
    if args.epochs is not None and (args.mode != "train" or args.epochs < 1 or not args.lora_rank):
        parser.error("--epochs requires LoRA training and a positive count")
    if args.resume and args.epochs is None:
        parser.error("--resume requires --epochs")
    base = root / "checkpoints/base"
    data = RoboICLDataset(
        args.work, "train" if args.mode == "train" else "validation", base / "vae",
        max_frames=args.max_frames if args.mode != "infer" else 2,
    )
    if args.mode == "inspect":
        sample = data[args.pair]
        print(json.dumps({"pairs": {k: len(v) for k, v in data.plan["pairs"].items()},
                          "held_out_pair": data.pairs[args.pair],
                          "sample_shapes": {k: list(v.shape) for k, v in sample.items()},
                          "first_group_masked": not sample["actions_mask"][:, 0].any().item()}))
        return
    if args.output is None:
        parser.error("--output is required")
    torch.manual_seed(args.seed + int(os.environ.get("RANK", "0")))

    from configs import VA_CONFIGS
    from easydict import EasyDict

    config = EasyDict(VA_CONFIGS["demo_train"].copy())
    config.update(
        wan22_pretrained_model_name_or_path=str(base),
        enable_wandb=False,
        height=224,
        width=288,
        obs_cam_keys=["external"],
        frame_chunk_size=2,
        attn_window=16,
        action_per_frame=ACTION_INTERVAL,
        used_action_channel_ids=list(ACTION_CHANNELS),
        norm_stat=data.norm_stat(),
        guidance_scale=1,
        action_guidance_scale=1,
        save_root=str(args.output),
    )
    if args.mode == "train":
        from wan_va import train
        from epoch_state import epoch_updates, latest_checkpoint

        world = int(os.environ.get("WORLD_SIZE", "1"))
        updates = epoch_updates(len(data), world, args.accumulation) if args.epochs else 0
        steps = updates * args.epochs if updates else args.steps
        metadata_path = args.output / "roboicl.json"
        settings = {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items() if key != "resume"}
        contract = {
            "plan_sha256": data.receipt["plan_sha256"],
            "stats_sha256": data.receipt["stats_sha256"],
            "source_rev": (root / "SOURCE_REV").read_text().strip(),
            "world_size": world,
            "updates_per_epoch": updates,
            "settings": settings,
        }
        restart = None
        if args.resume:
            if json.loads(metadata_path.read_text()) != contract:
                parser.error("resume settings or data differ from the original run")
            restart = latest_checkpoint(args.output)

        config.update(
            roboicl_work=str(args.work),
            roboicl_max_frames=args.max_frames,
            load_worker=args.workers,
            batch_size=1,
            num_steps=steps,
            save_interval=min(args.save_interval, steps),
            epoch_updates=updates,
            restart_directory=str(restart) if restart else None,
            gradient_accumulation_steps=args.accumulation,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_seed=args.seed,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate if args.learning_rate is not None
            else (1e-4 if args.lora_rank else 1e-5),
        )
        if int(os.environ.get("RANK", "0")) == 0 and not args.resume:
            args.output.mkdir(parents=True, exist_ok=False)
            temporary = metadata_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(contract, indent=2))
            temporary.replace(metadata_path)
        VA_CONFIGS["roboicl"] = config
        train.init_logger()
        train.run(SimpleNamespace(config_name="roboicl", save_root=str(args.output)))
        if int(os.environ.get("RANK", "0")) == 0:
            directory = args.output / "checkpoints" / f"checkpoint_step_{steps}"
            checkpoint = (directory / ADAPTER_FILE if args.lora_rank else
                          directory / "transformer/diffusion_pytorch_model.safetensors")
            if not checkpoint.is_file():
                raise RuntimeError(f"Expected final checkpoint at {checkpoint}")
        return

    if args.checkpoint is None:
        parser.error("infer requires --checkpoint pointing to checkpoint_step_N")
    adapter = args.checkpoint / ADAPTER_FILE
    transformer = args.checkpoint / "transformer"
    if not adapter.is_file() and not (transformer / "diffusion_pytorch_model.safetensors").is_file():
        parser.error(f"missing fine-tuned weights under {args.checkpoint}")
    run_metadata = json.loads((args.checkpoint.parent.parent / "roboicl.json").read_text())
    for key in ("plan_sha256", "stats_sha256"):
        if run_metadata[key] != data.receipt[key]:
            raise ValueError(f"Checkpoint data contract differs in {key}")

    import numpy as np
    from diffusers.utils import export_to_video
    from diffusers.video_processor import VideoProcessor
    from wan_va import wan_va_server as server

    class PairInference(server.VA_Server):
        def _encode_obs(self, obs):
            return obs["initial_latent"].to(device=self.device, dtype=self.dtype)

        def _repeat_input_for_cfg(self, inputs):
            inputs = super()._repeat_input_for_cfg(inputs)
            del inputs["text_emb"]
            inputs["demo_latents"] = self.demo_latents
            return inputs

    @torch.inference_mode()
    def infer():
        config.update(
            enable_offload=True,
            transformer_checkpoint=None if adapter.is_file() else str(transformer),
            lora_checkpoint=str(adapter) if adapter.is_file() else None,
            num_inference_steps=25,
            action_num_inference_steps=50,
        )
        args.output.mkdir(parents=True, exist_ok=False)
        server.init_logger()
        server.init_distributed(1, 0, 0)
        config.local_rank = 0
        model = PairInference(config)
        pair = data.pairs[args.pair]
        selected = pair["demo"] if args.demo_index is None else data.pairs[args.demo_index]["demo"]
        model.demo_latents = data.demo(selected).unsqueeze(0).to(model.device)
        model._reset()
        # The upstream input builder expects this field, which the visual context replaces.
        model.prompt_embeds = torch.empty((1, 0, 4096), device=model.device, dtype=model.dtype)
        initial = data.load_record("query", pair["query"])["latent"][:, :1].unsqueeze(0)
        actions, latents = model._infer({"initial_latent": initial})
        first_actions = actions[:, 1].T
        if first_actions.shape != (ACTION_INTERVAL, len(ACTION_CHANNELS)) or not np.isfinite(first_actions).all():
            raise RuntimeError("Inference did not produce finite 16×10 actions")
        np.save(args.output / "actions.npy", first_actions)
        (args.output / "pair.json").write_text(json.dumps({
            "held_out_pair": data.pairs[args.pair], "demo_used": selected, "seed": args.seed,
        }, indent=2))
        model.transformer.clear_cache(model.cache_name)
        del model.transformer
        torch.cuda.empty_cache()
        model.video_processor = VideoProcessor(vae_scale_factor=1)
        model.vae.to(device=model.device, dtype=model.dtype)
        video = model.decode_one_video(latents, "np")[0]
        export_to_video(video, str(args.output / "demo.mp4"), fps=12.5)
        torch.distributed.destroy_process_group()

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("infer requires one rank")
    infer()


if __name__ == "__main__":
    main()
