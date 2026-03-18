import unittest

import torch

from RECLUSE.moco import builder


class TestMoCoBuilderSingleProcess(unittest.TestCase):
    def test_concat_all_gather_returns_input_without_distributed_group(self):
        x = torch.randn(2, 3)
        y = builder.concat_all_gather(x)
        self.assertTrue(torch.equal(x, y))

    def test_contrastive_loss_runs_on_cpu_without_distributed_group(self):
        q = torch.randn(4, 8)
        k = torch.randn(4, 8)

        class DummyMoCo:
            T = 1.0

        loss = builder.MoCo.contrastive_loss(DummyMoCo(), q, k)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss).item())
