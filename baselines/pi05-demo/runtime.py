"""Uniform Query-frame indexing with RGB-only Demo selection from frozen pairs."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler

from data import read, write, sha
from openpi.models.tokenizer import PaligemmaTokenizer


class RemainingBatches(BatchSampler):
    start = 0

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.start, None)


class FrameLoader:
    """The standard torch distributed sampler/DataLoader, with a saved epoch cursor."""
    dynamic_graph = True  # Demo masks and last-window valid lengths vary by batch.

    def __iter__(self):
        self.dataset.epoch = self.epoch
        self.sampler.set_epoch(self.epoch)
        self.batches.start = self.cursor
        batches=iter(self.loader)
        if getattr(self,'prefetch_depth',0):
            from fast_execution import prefetch
            batches=prefetch(batches,self.prefetch_depth)
        for batch in batches:
            self.cursor += 1
            yield batch
        self.epoch += 1
        self.cursor = 0

    def state_dict(self):
        return dict(epoch=self.epoch, batches=self.cursor, frames=len(self.dataset),
                    global_batch=self.global_batch, world=self.world, sampler='uniform_query_frames_v1')

    def load_state_dict(self, state):
        expected = self.state_dict()
        assert all(state[k] == expected[k] for k in ('frames', 'global_batch', 'world', 'sampler'))
        assert 0 <= state['batches'] <= len(self.batches)
        self.epoch, self.cursor = state['epoch'], state['batches']

    def record_update(self, model, step, loss, grad_norm, lr, seconds, output):
        values = torch.tensor([loss.detach().item(), float(grad_norm), seconds,
                               torch.cuda.max_memory_allocated()], device=loss.device, dtype=torch.float64)
        if dist.is_initialized():
            avg = values[:1].clone(); dist.all_reduce(avg); values[0] = avg[0] / self.world
            dist.all_reduce(values[1:], op=dist.ReduceOp.MAX)
        if self.rank == 0:
            row = dict(step=step, loss=float(values[0]), gradient_norm=float(values[1]), lr=lr,
                       seconds=float(values[2]), peak_allocated_bytes=int(values[3]),
                       global_batch=self.global_batch, accumulation=1, sampler=self.state_dict())
            row['branch_gradients'] = self.gradient_receipt
            with (output/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')

    def write_checkpoint_receipt(self, folder, step):
        write(folder/'complete.json', dict(step=step, contract=self.contract,
              model_sha256=sha(folder/'model.safetensors'), optimizer_sha256=sha(folder/'optimizer.pt'),
              metadata_sha256=sha(folder/'metadata.pt'), checkpoint_format='pi05_demo_lora_v1'))

    def verify_checkpoint(self, folder):
        marker = read(folder/'complete.json')
        for name,key in [('model.safetensors','model_sha256'),('optimizer.pt','optimizer_sha256'),('metadata.pt','metadata_sha256')]:
            assert sha(folder/name)==marker[key],name
        assert marker['contract']==self.contract
