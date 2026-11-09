"""CPU smoke tests for the epoch and adapter contracts."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from epoch_state import EpochSampler, epoch_updates, local_optimizer_state, restore_optimizer
from lora import ADAPTER_FILE, TARGETS, install_lora, load_lora, merge_lora, save_lora


class WorkflowTest(unittest.TestCase):
    def test_epoch_and_resume_cursor(self):
        world, accumulation, pairs = 4, 16, 21860
        self.assertEqual(epoch_updates(pairs, world, accumulation), 341)
        samplers = [EpochSampler(pairs, world, rank, accumulation, 42) for rank in range(world)]
        samples = [index for sampler in samplers for index in sampler]
        self.assertEqual(len(set(samples)), 21824)
        for sampler in samplers:
            original = list(sampler)
            sampler.seek(50, accumulation)
            self.assertEqual(list(sampler), original[800:])
            sampler.seek(341, accumulation)
            self.assertEqual(sampler.epoch, 1)
            self.assertNotEqual(list(sampler), original)

    def test_adapter_update_restart_and_inference(self):
        torch.manual_seed(1)
        base = nn.Module()
        block = nn.Module()
        block.attn2 = nn.Module()
        for target in TARGETS:
            if target == "to_out.0":
                block.attn2.to_out = nn.ModuleList([nn.Linear(4, 4)])
            else:
                setattr(block.attn2, target, nn.Linear(4, 4))
        base.blocks = nn.ModuleList([block])
        trained, resumed = deepcopy(base), deepcopy(base)
        inputs = torch.randn(2, 4)

        def output(model):
            return model.blocks[0].attn2.to_q(inputs)

        def optimizer(model):
            return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])

        def update(model, opt):
            opt.zero_grad()
            output(model).square().mean().backward()
            opt.step()

        install_lora(trained, 2, 4, 42)
        install_lora(resumed, 2, 4, 42)
        torch.testing.assert_close(output(trained), output(base))
        train_opt, resume_opt = optimizer(trained), optimizer(resumed)
        update(trained, train_opt)
        self.assertFalse(torch.equal(output(trained), output(base)))
        self.assertTrue(all(p.grad is None for p in trained.parameters() if not p.requires_grad))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            save_lora(trained, path, 2, 4, True)
            load_lora(resumed, path / ADAPTER_FILE, 2, 4)
            torch.save(local_optimizer_state(train_opt), path / "optim.pt")
            restore_optimizer(resume_opt, torch.load(path / "optim.pt", weights_only=True))
            merge_lora(base, path / ADAPTER_FILE)
        torch.testing.assert_close(output(trained), output(base))
        update(trained, train_opt)
        update(resumed, resume_opt)
        torch.testing.assert_close(output(trained), output(resumed), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
