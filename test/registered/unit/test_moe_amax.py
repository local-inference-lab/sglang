import atexit
import os
import tempfile
import unittest

import torch
from safetensors.torch import load_file

from sglang.srt.utils.moe_amax import MoEAmaxTracker
from sglang.test.test_utils import CustomTestCase


def _unregister_flush(tracker: MoEAmaxTracker):
    atexit.unregister(tracker.flush)


class TestMoEAmaxTracker(CustomTestCase):
    def test_tracks_prefixed_moe_input_amax(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "amax.safetensors")
            tracker = MoEAmaxTracker(output_path=output_path, write_interval=1)
            self.addCleanup(_unregister_flush, tracker)

            hidden_states = torch.tensor(
                [
                    [1.0, -3.0],
                    [10.0, 4.0],
                    [5.0, -6.0],
                ]
            )
            topk_ids = torch.tensor(
                [
                    [0, 1],
                    [1, -1],
                    [3, 2],
                ],
                dtype=torch.int32,
            )

            tracker.update_moe_inputs(
                module_prefix="model.layers.7.mlp",
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                num_routed_experts=3,
            )
            tracker.flush()

            tensors = load_file(output_path)
            expected = {
                "model.layers.7.mlp.experts.0.gate_proj.input_quantizer": 3.0,
                "model.layers.7.mlp.experts.0.up_proj.input_quantizer": 3.0,
                "model.layers.7.mlp.experts.1.gate_proj.input_quantizer": 10.0,
                "model.layers.7.mlp.experts.1.up_proj.input_quantizer": 10.0,
                "model.layers.7.mlp.experts.2.gate_proj.input_quantizer": 6.0,
                "model.layers.7.mlp.experts.2.up_proj.input_quantizer": 6.0,
            }
            self.assertEqual(set(tensors), set(expected))
            for name, value in expected.items():
                self.assertEqual(tensors[name].item(), value)


if __name__ == "__main__":
    unittest.main()
