"""Read directed RoboICL pairs using the verified Wan video-latent cache."""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ACTION_CHANNELS = (*range(9), 28)
ACTION_INTERVAL = 16
DEMO_FRAMES = 8
DEMO_SIZE = 8


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


class RoboICLDataset(Dataset):
    def __init__(self, work, split, vae, max_frames=None):
        self.work = Path(work)
        self.max_frames = max_frames
        self.receipt = json.loads((self.work / "verified-cache.json").read_text())
        plan_file = self.work / "plan.json"
        stats_file = self.work / "stats.json"
        for path in (plan_file, stats_file):
            if digest(path) != self.receipt[f"{path.stem}_sha256"]:
                raise ValueError(f"Cache receipt does not match {path}")
        self.plan = json.loads(plan_file.read_text())
        self.stats = json.loads(stats_file.read_text())
        contract = self.receipt["contract"]
        if (contract["action_interval"], contract["encoding"], self.stats["representation"]) != (
            ACTION_INTERVAL, "wan_causal_external_rgb_bf16_mu_v1", "roboicl_world_hand10_absolute"
        ):
            raise ValueError("Unexpected RoboICL cache contract")
        for filename, expected in contract["encoder"].items():
            if digest(Path(vae) / filename) != expected:
                raise ValueError(f"RoboICL cache was encoded with a different VAE: {filename}")
        self.pairs = self.plan["pairs"][split]
        if not self.pairs:
            raise ValueError(f"Empty RoboICL split: {split}")
        self.lower = torch.tensor(self.stats["action"]["q01"])
        self.upper = torch.tensor(self.stats["action"]["q99"])

    def __len__(self):
        return len(self.pairs)

    def load_record(self, role, episode):
        relative = f"latents/{role}/{episode}.pt"
        path = self.work / relative
        stat = path.stat()
        if self.receipt["latents"][relative] != [stat.st_size, stat.st_mtime_ns]:
            raise ValueError(f"RoboICL cache changed since verification: {path}")
        payload = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        if (payload["recording"], payload["role"]) != (episode, role):
            raise ValueError(f"Unexpected RoboICL cache identity: {path}")
        return payload

    def demo(self, episode):
        latent = self.load_record("demo", episode)["latent"]
        indices = torch.linspace(0, latent.shape[1] - 1, DEMO_FRAMES).round().long()
        frames = latent[:, indices].transpose(0, 1).float()
        pooled = F.adaptive_avg_pool2d(frames, (DEMO_SIZE, DEMO_SIZE))
        return pooled.transpose(0, 1).contiguous().to(torch.bfloat16)

    def __getitem__(self, index):
        pair = self.pairs[index]
        if pair["demo"] == pair["query"]:
            raise ValueError("RoboICL pair contains the same recording twice")
        query = self.load_record("query", pair["query"])
        latent = query["latent"][:, :self.max_frames]
        frames = latent.shape[1]
        raw = query["actions"].float()
        anchors = query["latent_nominal_anchor_indices"][:frames]
        if frames < 2 or not torch.equal(anchors, torch.arange(frames) * ACTION_INTERVAL):
            raise ValueError("Unexpected query latent anchors")
        ids = anchors[:, None] - ACTION_INTERVAL + torch.arange(ACTION_INTERVAL)
        valid = (ids >= 0) & (ids < len(raw))
        values = raw[ids.clamp(0, len(raw) - 1)]
        values = (2 * (values - self.lower) / (self.upper - self.lower + 1e-6) - 1)
        values = values.clamp(-self.stats["clip"], self.stats["clip"])
        actions = torch.zeros(30, frames, ACTION_INTERVAL, 1)
        mask = torch.zeros_like(actions, dtype=torch.bool)
        channels = list(ACTION_CHANNELS)
        actions[channels, :, :, 0] = (values * valid[..., None]).permute(2, 0, 1)
        mask[channels, :, :, 0] = valid
        return {
            "latents": latent.contiguous(),
            "actions": actions,
            "actions_mask": mask,
            "demo_latents": self.demo(pair["demo"]),
        }

    def norm_stat(self):
        lower = torch.zeros(30)
        upper = torch.zeros(30)
        lower[list(ACTION_CHANNELS)] = self.lower
        upper[list(ACTION_CHANNELS)] = self.upper
        return {"q01": lower.tolist(), "q99": upper.tolist()}
