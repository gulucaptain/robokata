"""Match a video frame's semantic key against the demonstration frames."""

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_ifp import pool_video_states


# The read layer must follow the layer that supplies its video key.
KEY_LAYERS = {3: 7, 11: 15, 19: 23}
SCHEMA = "dino_frame_index_v1"


class ForesightIndexing(nn.Module):
    def __init__(self, hidden_dim, target_dim, grid, demo_frames, width=256, *, views=1, query_views=None):
        super().__init__()
        self.target_dim = target_dim
        self.grid = grid
        self.views = views
        self.query_views = views if query_views is None else query_views
        if (self.views, self.query_views) not in ((1, 1), (2, 2), (1, 2)):
            raise ValueError("Unsupported demo/query camera combination")
        self.demo_frames = demo_frames
        self.key_heads = nn.ModuleDict({
            str(layer): nn.Sequential(
                nn.Linear(hidden_dim, width), nn.GELU(), nn.Linear(width, target_dim)
            )
            for layer in KEY_LAYERS
        })
        self.projection = nn.Linear(target_dim, width, bias=False)
        # Keep scalar parameters one-dimensional for FSDP2.
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def video_key(self, layer, states, frames, height, width):
        pooled = pool_video_states(states, frames, height, width, self.grid, self.query_views)
        return self.key_heads[str(layer)](pooled)

    def log_index(self, key, demo_tokens):
        # Frame and cell one-hots identify the attention tokens, but matching
        # compares only the DINO features at corresponding spatial cells.
        demo = demo_tokens[..., :self.target_dim].unflatten(1, (self.demo_frames, self.views * self.grid**2))
        # Camera 0 is top view. The key loss still supervises all query cameras.
        matching_key = key[..., :self.views * self.grid**2, :]
        query = F.normalize(self.projection(matching_key.detach()).float(), dim=-1)
        demo = F.normalize(self.projection(demo).float(), dim=-1)
        scores = torch.einsum("bfgd,bkgd->bfk", query, demo) / (self.views * self.grid**2)
        temperature = self.log_temperature.float().clamp(-5, 5).exp()
        return F.log_softmax(scores / temperature, dim=-1)


def key_loss(keys, features, valid):
    """Sum the cosine regression losses over key layers on recorded frames."""
    target = features.detach().float()
    mask = valid.unsqueeze(-1).expand_as(features[..., 0])
    count = mask.sum().clamp_min(1)
    return sum(
        (1 - F.cosine_similarity(key.float(), target, dim=-1)).masked_fill(~mask, 0).sum() / count
        for key in keys
    )
