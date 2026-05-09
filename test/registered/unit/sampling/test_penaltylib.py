"""Unit tests for srt/sampling/penaltylib/ — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="stage-a-test-cpu")

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.sampling.penaltylib.entropy_penalty import (
    BatchedEntropyPenalizer,
)
from sglang.srt.sampling.penaltylib.frequency_penalty import (
    BatchedFrequencyPenalizer,
)
from sglang.srt.sampling.penaltylib.min_new_tokens import (
    BatchedMinNewTokensPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
)
from sglang.srt.sampling.penaltylib.presence_penalty import (
    BatchedPresencePenalizer,
)
from sglang.test.test_utils import CustomTestCase

VOCAB_SIZE = 32
DEVICE = "cpu"


# Helpers: mock Req and ScheduleBatch
def _make_req(
    freq=0.0,
    presence=0.0,
    min_tokens=0,
    stop_ids=None,
    eos_id=2,
    entropy=0.0,
    entropy_min_len=16,
    entropy_max_len=256,
    entropy_window=8192,
    entropy_max_penalty=8.0,
    entropy_min_repetitions=1,
):
    """Create a mock request with sampling params."""
    req = MagicMock()
    req.sampling_params.frequency_penalty = freq
    req.sampling_params.presence_penalty = presence
    req.sampling_params.min_new_tokens = min_tokens
    req.sampling_params.entropy_penalty = entropy
    req.sampling_params.entropy_penalty_min_len = entropy_min_len
    req.sampling_params.entropy_penalty_max_len = entropy_max_len
    req.sampling_params.entropy_penalty_window = entropy_window
    req.sampling_params.entropy_penalty_max_penalty = entropy_max_penalty
    req.sampling_params.entropy_penalty_min_repetitions = entropy_min_repetitions
    req.sampling_params.stop_token_ids = stop_ids
    req.tokenizer.additional_stop_token_ids = None
    req.tokenizer.eos_token_id = eos_id
    req.output_ids = []
    req.origin_input_ids = []
    return req


def _make_batch(reqs):
    """Create a mock ScheduleBatch.
    Note: orchestrator accesses batch.reqs as an attribute (not a method call)."""
    batch = MagicMock()
    batch.reqs = reqs
    batch.device = DEVICE
    return batch


# BatchedPenalizerOrchestrator
class TestBatchedPenalizerOrchestrator(CustomTestCase):
    def test_init_detects_required_penalizers(self):
        """Test that orchestrator marks is_required=True when any request has nonzero penalty."""
        reqs = [_make_req(freq=1.0)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        self.assertTrue(orch.is_required)

    def test_init_not_required_when_no_penalties(self):
        """Test that orchestrator marks is_required=False when all penalties are zero."""
        reqs = [_make_req()]  # all defaults (0.0)
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        self.assertFalse(orch.is_required)

    def test_batch_property_via_weakref(self):
        """Test that batch property returns the original batch via weakref."""
        reqs = [_make_req()]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(VOCAB_SIZE, batch, set())
        self.assertIs(orch.batch, batch)

    def test_batch_setter_none(self):
        """Test that setting batch to None breaks the weakref cleanly."""
        reqs = [_make_req()]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(VOCAB_SIZE, batch, set())
        orch.batch = None
        self.assertIsNone(orch.batch)

    def test_batch_setter_new_batch(self):
        """Test that batch can be reassigned to a different ScheduleBatch."""
        reqs = [_make_req()]
        batch1 = _make_batch(reqs)
        batch2 = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(VOCAB_SIZE, batch1, set())
        orch.batch = batch2
        self.assertIs(orch.batch, batch2)

    def test_context_manager_releases(self):
        """Test that exiting the context manager releases all penalizers."""
        reqs = [_make_req(freq=1.0)]
        batch = _make_batch(reqs)
        with BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        ) as orch:
            self.assertTrue(orch.is_required)
        self.assertFalse(orch.is_required)
        self.assertEqual(len(orch.penalizers), 0)

    def test_filter_empty_indices_releases(self):
        """Test that filtering with no indices left fully releases the orchestrator."""
        reqs = [_make_req(freq=1.0)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        orch.filter(torch.tensor([], dtype=torch.long))
        self.assertFalse(orch.is_required)

    def test_filter_not_required_is_noop(self):
        """Test that filter on a not-required orchestrator does nothing."""
        reqs = [_make_req()]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        self.assertFalse(orch.is_required)
        orch.filter(torch.tensor([0]))  # should not raise

    def test_merge_both_not_required_is_noop(self):
        """Test that merging two not-required orchestrators stays not-required."""
        reqs = [_make_req()]
        batch = _make_batch(reqs)
        orch1 = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        orch2 = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        orch1.merge(orch2)  # should not raise
        self.assertFalse(orch1.is_required)


# BatchedFrequencyPenalizer
class TestBatchedFrequencyPenalizer(CustomTestCase):
    def _setup(self, freq_values):
        reqs = [_make_req(freq=f) for f in freq_values]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        pen = orch.penalizers[BatchedFrequencyPenalizer]
        return orch, pen

    def test_is_required_with_nonzero_penalty(self):
        """Test that nonzero frequency_penalty makes the penalizer required."""
        _, pen = self._setup([1.5])
        self.assertTrue(pen.is_required())

    def test_is_not_required_with_zero_penalty(self):
        """Test that zero frequency_penalty makes the penalizer not required."""
        _, pen = self._setup([0.0])
        self.assertFalse(pen.is_required())

    def test_cumulate_and_apply(self):
        """Test that cumulating a token applies frequency penalty to its logit."""
        orch, pen = self._setup([2.0])
        output_ids = torch.tensor([5])
        pen.cumulate_output_tokens(output_ids)

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertAlmostEqual(logits[0, 5].item(), -2.0, places=5)
        # Other tokens unaffected
        self.assertAlmostEqual(logits[0, 0].item(), 0.0, places=5)

    def test_cumulate_twice_doubles_penalty(self):
        """Test that frequency penalty scales linearly with occurrence count."""
        orch, pen = self._setup([1.0])
        pen.cumulate_output_tokens(torch.tensor([3]))
        pen.cumulate_output_tokens(torch.tensor([3]))

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertAlmostEqual(logits[0, 3].item(), -2.0, places=5)

    def test_filter_keeps_subset(self):
        """Test that filter retains only the selected batch indices."""
        orch, pen = self._setup([1.0, 2.0])
        keep = torch.tensor([1])
        pen.filter(keep)
        self.assertEqual(pen.frequency_penalties.shape[0], 1)
        self.assertAlmostEqual(pen.frequency_penalties[0, 0].item(), 2.0, places=5)

    def test_merge_concatenates(self):
        """Test that merge concatenates penalty tensors from two penalizers."""
        _, pen1 = self._setup([1.0])
        _, pen2 = self._setup([2.0])
        pen1.merge(pen2)
        self.assertEqual(pen1.frequency_penalties.shape[0], 2)

    def test_teardown_cleans_attributes(self):
        """Test that teardown deletes internal tensors and resets prepared state."""
        _, pen = self._setup([1.0])
        pen.teardown()
        self.assertFalse(hasattr(pen, "frequency_penalties"))
        self.assertFalse(hasattr(pen, "cumulated_frequency_penalties"))
        self.assertFalse(pen.is_prepared())

    def test_cumulate_when_not_prepared_is_noop(self):
        """Test that cumulate before prepare does not crash."""
        _, pen = self._setup([0.0])
        # pen is not prepared (is_required=False)
        pen.cumulate_output_tokens(torch.tensor([1]))  # should not raise

    def test_apply_when_not_prepared_is_noop(self):
        """Test that apply on an unprepared penalizer leaves logits unchanged."""
        _, pen = self._setup([0.0])
        logits = torch.zeros(1, VOCAB_SIZE)
        original = logits.clone()
        pen.apply(logits)
        self.assertTrue(torch.equal(logits, original))


# BatchedPresencePenalizer
class TestBatchedPresencePenalizer(CustomTestCase):
    def _setup(self, presence_values):
        reqs = [_make_req(presence=p) for p in presence_values]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedPresencePenalizer}
        )
        pen = orch.penalizers[BatchedPresencePenalizer]
        return orch, pen

    def test_is_required_with_nonzero_penalty(self):
        """Test that nonzero presence_penalty makes the penalizer required."""
        _, pen = self._setup([0.5])
        self.assertTrue(pen.is_required())

    def test_presence_penalty_does_not_scale(self):
        """Test that presence penalty is flat (same value regardless of count)."""
        orch, pen = self._setup([1.0])
        pen.cumulate_output_tokens(torch.tensor([7]))
        pen.cumulate_output_tokens(torch.tensor([7]))  # same token again

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        # scatter_ overwrites (not adds), so penalty should be 1.0, not 2.0
        self.assertAlmostEqual(logits[0, 7].item(), -1.0, places=5)

    def test_filter_keeps_subset(self):
        """Test that filter retains the first request's presence penalty."""
        orch, pen = self._setup([1.0, 2.0])
        keep = torch.tensor([0])
        pen.filter(keep)
        self.assertEqual(pen.presence_penalties.shape[0], 1)
        self.assertAlmostEqual(pen.presence_penalties[0, 0].item(), 1.0, places=5)

    def test_merge_concatenates(self):
        """Test that merge concatenates presence penalty tensors."""
        _, pen1 = self._setup([1.0])
        _, pen2 = self._setup([2.0])
        pen1.merge(pen2)
        self.assertEqual(pen1.presence_penalties.shape[0], 2)

    def test_teardown_cleans_attributes(self):
        """Test that teardown removes the presence_penalties tensor."""
        _, pen = self._setup([1.0])
        pen.teardown()
        self.assertFalse(hasattr(pen, "presence_penalties"))


# BatchedMinNewTokensPenalizer
class TestBatchedMinNewTokensPenalizer(CustomTestCase):
    def _setup(self, configs):
        """configs: list of (min_tokens, stop_ids, eos_id)."""
        reqs = [_make_req(min_tokens=c[0], stop_ids=c[1], eos_id=c[2]) for c in configs]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedMinNewTokensPenalizer}
        )
        pen = orch.penalizers[BatchedMinNewTokensPenalizer]
        return orch, pen

    def test_is_required_with_positive_min_tokens(self):
        """Test that positive min_new_tokens makes the penalizer required."""
        _, pen = self._setup([(5, None, 2)])
        self.assertTrue(pen.is_required())

    def test_is_not_required_with_zero_min_tokens(self):
        """Test that min_new_tokens=0 makes the penalizer not required."""
        _, pen = self._setup([(0, None, 2)])
        self.assertFalse(pen.is_required())

    def test_blocks_eos_before_min_tokens(self):
        """Test that EOS token is blocked before min_new_tokens is reached."""
        orch, pen = self._setup([(3, None, 2)])
        # Before any output: len=0 < min=3 → block EOS (token 2)
        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertTrue(torch.isinf(logits[0, 2]) and logits[0, 2] < 0)
        # Non-stop tokens should be fine
        self.assertEqual(logits[0, 0].item(), 0.0)

    def test_allows_eos_after_min_tokens(self):
        """Test that EOS is allowed after generating min_new_tokens."""
        orch, pen = self._setup([(2, None, 2)])
        # Generate 2 tokens
        pen.cumulate_output_tokens(torch.tensor([10]))
        pen.cumulate_output_tokens(torch.tensor([11]))
        # Now len=2 >= min=2 → EOS should NOT be blocked
        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertEqual(logits[0, 2].item(), 0.0)

    def test_blocks_custom_stop_tokens(self):
        """Test that custom stop_token_ids are also blocked before min_new_tokens."""
        orch, pen = self._setup([(3, {5, 10}, 2)])
        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        # EOS (2), stop token 5, stop token 10 should all be blocked
        self.assertTrue(torch.isinf(logits[0, 2]) and logits[0, 2] < 0)
        self.assertTrue(torch.isinf(logits[0, 5]) and logits[0, 5] < 0)
        self.assertTrue(torch.isinf(logits[0, 10]) and logits[0, 10] < 0)

    def test_blocks_additional_stop_tokens(self):
        """Test that tokenizer's additional_stop_token_ids are also blocked."""
        req = _make_req(min_tokens=3, stop_ids=None, eos_id=2)
        req.tokenizer.additional_stop_token_ids = {7, 8}
        batch = _make_batch([req])
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedMinNewTokensPenalizer}
        )
        pen = orch.penalizers[BatchedMinNewTokensPenalizer]

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        # EOS (2) + additional stops (7, 8) should all be blocked
        for tok in [2, 7, 8]:
            self.assertTrue(
                torch.isinf(logits[0, tok]) and logits[0, tok] < 0,
                f"token {tok} should be blocked before min_new_tokens",
            )
        # Non-stop tokens should be fine
        self.assertEqual(logits[0, 0].item(), 0.0)

    def test_filter_keeps_subset(self):
        """Test that filter keeps the second request (min_tokens=5) and drops the first."""
        orch, pen = self._setup([(3, None, 2), (5, None, 2)])
        keep = torch.tensor([1])
        pen.filter(keep)
        self.assertEqual(pen.min_new_tokens.shape[0], 1)
        self.assertEqual(pen.min_new_tokens[0, 0].item(), 5)

    def test_merge_concatenates(self):
        """Test that merge combines min_new_tokens tensors from two penalizers."""
        _, pen1 = self._setup([(3, None, 2)])
        _, pen2 = self._setup([(5, None, 2)])
        pen1.merge(pen2)
        self.assertEqual(pen1.min_new_tokens.shape[0], 2)

    def test_teardown_cleans_attributes(self):
        """Test that teardown removes min_new_tokens, stop_token_penalties, and len_output_tokens."""
        _, pen = self._setup([(3, None, 2)])
        pen.teardown()
        self.assertFalse(hasattr(pen, "min_new_tokens"))
        self.assertFalse(hasattr(pen, "stop_token_penalties"))
        self.assertFalse(hasattr(pen, "len_output_tokens"))


# BatchedEntropyPenalizer
class TestBatchedEntropyPenalizer(CustomTestCase):
    def _setup(self, configs):
        """configs: list of entropy sampling param dictionaries."""
        reqs = [_make_req(**config) for config in configs]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedEntropyPenalizer}
        )
        orch._test_batch = batch
        pen = orch.penalizers[BatchedEntropyPenalizer]
        return orch, pen

    def _feed(self, pen, tokens):
        for token_id in tokens:
            pen.cumulate_output_tokens(torch.tensor([token_id]))

    def test_is_required_with_nonzero_penalty(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 1.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 5,
                }
            ]
        )
        self.assertTrue(pen.is_required())

    def test_is_not_required_with_zero_penalty(self):
        _, pen = self._setup([{"entropy": 0.0}])
        self.assertFalse(pen.is_required())

    def test_no_penalty_before_min_len(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 5,
                }
            ]
        )
        self._feed(pen, [1, 2])

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))

    def test_repeated_continuation_token_gets_penalty(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 5,
                    "entropy_max_penalty": 8.0,
                }
            ]
        )
        self._feed(pen, [1, 2, 3, 4, 1, 2, 3])

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        expected = 2.0
        self.assertAlmostEqual(logits[0, 4].item(), -expected, places=5)
        self.assertEqual(logits[0, 5].item(), 0.0)

    def test_default_like_settings_penalize_first_repeated_16_token_span(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 16,
                    "entropy_max_len": 512,
                    "entropy_max_penalty": 512.0,
                }
            ]
        )
        repeated_prefix = list(range(1, 17))
        self._feed(pen, repeated_prefix + [17] + repeated_prefix)

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertAlmostEqual(logits[0, 17].item(), -2.0, places=5)

    def test_default_like_settings_use_sparse_lz_index_lengths(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 16,
                    "entropy_max_len": 512,
                }
            ]
        )
        lengths = pen.states[0].indexed_match_lengths

        self.assertEqual(lengths[0], 16)
        self.assertEqual(lengths[-1], 512)
        self.assertLessEqual(len(lengths), 32)

    def test_default_like_settings_make_long_repeated_context_extreme(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 16,
                    "entropy_max_len": 512,
                    "entropy_max_penalty": 512.0,
                }
            ]
        )
        repeated_block = [(i % 30) + 1 for i in range(512)]
        self._feed(pen, (repeated_block + [31]) * 7 + repeated_block)

        details = pen.states[0].penalty_details()
        self.assertIn(31, details)
        self.assertEqual(details[31].match_len, 512)
        self.assertEqual(details[31].repeat_count, 7)
        self.assertAlmostEqual(details[31].penalty, 448.0, places=5)

    def test_longer_repeated_span_has_stronger_penalty(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 6,
                }
            ]
        )
        self._feed(pen, [1, 2, 3, 4, 9, 1, 2, 3])

        logits_short = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits_short)
        short_penalty = -logits_short[0, 4].item()

        self._feed(pen, [4])
        logits_long = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits_long)
        long_penalty = -logits_long[0, 9].item()

        self.assertGreater(short_penalty, 0.0)
        self.assertGreater(long_penalty, short_penalty)

    def test_periodic_repetition_shorter_than_min_len_gets_penalty(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 1.0,
                    "entropy_min_len": 9,
                    "entropy_max_len": 12,
                    "entropy_max_penalty": 8.0,
                }
            ]
        )
        self._feed(pen, [1, 2, 3] * 3)

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertAlmostEqual(logits[0, 1].item(), -3.0, places=5)
        self.assertEqual(logits[0, 4].item(), 0.0)

    def test_long_period_repetition_penalty_is_continuous(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 8,
                    "entropy_max_len": 24,
                    "entropy_max_penalty": 512.0,
                }
            ]
        )
        repeated_block = list(range(1, 21))
        self._feed(pen, repeated_block * 2)

        for expected_token in repeated_block:
            details = pen.states[0].penalty_details()
            self.assertIn(expected_token, details)
            self.assertGreaterEqual(details[expected_token].match_len, 8)
            self.assertGreater(details[expected_token].penalty, 0.0)
            pen.cumulate_output_tokens(torch.tensor([expected_token]))

    def test_applied_log_uses_fixed_interval_not_exponential_backoff(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 5,
                }
            ]
        )
        self._feed(pen, [1, 2, 3, 4, 1, 2, 3])
        pen.states[0].logged_applications = 383

        logits = torch.zeros(1, VOCAB_SIZE)
        with self.assertLogs(
            "sglang.srt.sampling.penaltylib.entropy_penalty", level="INFO"
        ) as log:
            pen.apply(logits)

        self.assertIn("application_count=384", "\n".join(log.output))

    def test_repeat_count_increases_penalty(self):
        _, pen_one = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 5,
                }
            ]
        )
        self._feed(pen_one, [1, 2, 3, 4, 8, 9, 1, 2, 3])
        logits_one = torch.zeros(1, VOCAB_SIZE)
        pen_one.apply(logits_one)

        _, pen_two = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 5,
                }
            ]
        )
        self._feed(pen_two, [1, 2, 3, 4, 8, 9, 1, 2, 3, 4, 10, 11, 1, 2, 3])
        logits_two = torch.zeros(1, VOCAB_SIZE)
        pen_two.apply(logits_two)

        self.assertGreater(-logits_two[0, 4].item(), -logits_one[0, 4].item())
        expected = 4.0
        self.assertAlmostEqual(logits_two[0, 4].item(), -expected, places=5)

    def test_min_repetitions_suppresses_low_repeat_lz_candidates(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 5,
                    "entropy_min_repetitions": 2,
                }
            ]
        )
        self._feed(pen, [1, 2, 3, 4, 8, 9, 1, 2, 3])

        logits_one = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits_one)
        self.assertEqual(logits_one[0, 4].item(), 0.0)

        self._feed(pen, [4, 10, 11, 1, 2, 3])
        logits_two = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits_two)
        self.assertAlmostEqual(logits_two[0, 4].item(), -4.0, places=5)

    def test_min_repetitions_suppresses_low_repeat_periodic_candidates(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 1.0,
                    "entropy_min_len": 9,
                    "entropy_max_len": 12,
                    "entropy_min_repetitions": 4,
                }
            ]
        )
        self._feed(pen, [1, 2, 3] * 3)

        logits_three = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits_three)
        self.assertEqual(logits_three[0, 1].item(), 0.0)

        self._feed(pen, [1, 2, 3])
        logits_four = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits_four)
        self.assertAlmostEqual(logits_four[0, 1].item(), -(12 / 9) * 4, places=5)

    def test_repeated_larger_block_hits_cap(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 4,
                    "entropy_max_len": 12,
                    "entropy_max_penalty": 6.0,
                }
            ]
        )
        paragraph = [1, 2, 3, 4]
        self._feed(pen, paragraph * 4)

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertAlmostEqual(logits[0, 1].item(), -6.0, places=5)

    def test_penalty_is_capped(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 100.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 5,
                    "entropy_max_penalty": 0.25,
                }
            ]
        )
        self._feed(pen, [1, 2, 3, 4, 1, 2, 3])

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertAlmostEqual(logits[0, 4].item(), -0.25, places=5)

    def test_window_trimming_drops_old_continuations(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 2,
                    "entropy_max_len": 2,
                    "entropy_window": 2,
                }
            ]
        )
        self._feed(pen, [1, 2, 3, 9, 9, 9, 1, 2])

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertEqual(logits[0, 3].item(), 0.0)

    def test_window_trimming_does_not_rebuild_index_per_token(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 4,
                    "entropy_max_len": 4,
                    "entropy_window": 4,
                }
            ]
        )
        state = pen.states[0]
        rebuild_calls = 0
        original_rebuild = state._rebuild

        def counting_rebuild():
            nonlocal rebuild_calls
            rebuild_calls += 1
            original_rebuild()

        state._rebuild = counting_rebuild
        self._feed(pen, [token_id % VOCAB_SIZE for token_id in range(64)])

        self.assertEqual(rebuild_calls, 0)
        self.assertLess(len(state.history), state.trim_threshold_len)

    def test_filter_keeps_matching_state(self):
        _, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 5,
                },
                {
                    "entropy": 2.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 5,
                },
            ]
        )
        for token_id in [1, 2, 3]:
            pen.states[0].append(token_id)
        for token_id in [4, 5, 6, 7, 4, 5, 6]:
            pen.states[1].append(token_id)

        pen.filter(torch.tensor([1]))
        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertLess(logits[0, 7].item(), 0.0)

    def test_merge_concatenates_states(self):
        _, pen1 = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 5,
                }
            ]
        )
        _, pen2 = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 3,
                    "entropy_max_len": 5,
                }
            ]
        )
        for token_id in [4, 5, 6, 7, 4, 5, 6]:
            pen2.states[0].append(token_id)

        pen1.merge(pen2)
        logits = torch.zeros(2, VOCAB_SIZE)
        pen1.apply(logits)
        self.assertEqual(logits[0, 7].item(), 0.0)
        self.assertLess(logits[1, 7].item(), 0.0)

    def test_overlap_prompt_token_is_not_counted_as_output(self):
        orch, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 1,
                    "entropy_max_len": 1,
                }
            ]
        )
        orch.batch.reqs[0].origin_input_ids = [10]
        orch.batch.reqs[0].output_ids = []

        pen.cumulate_output_tokens(torch.tensor([10]))
        self.assertEqual(pen.states[0].history, [])

        orch.batch.reqs[0].output_ids = [10]
        pen.cumulate_output_tokens(torch.tensor([10]))
        orch.batch.reqs[0].output_ids = [10, 10]
        pen.cumulate_output_tokens(torch.tensor([10]))
        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertLess(logits[0, 10].item(), 0.0)

    def test_speculative_accepts_cumulate_all_new_output_ids(self):
        orch, pen = self._setup(
            [
                {
                    "entropy": 2.0,
                    "entropy_min_len": 1,
                    "entropy_max_len": 1,
                }
            ]
        )
        req = orch.batch.reqs[0]

        req.output_ids = [1, 2, 3, 4]
        pen.cumulate_output_tokens(torch.tensor([4]))
        self.assertEqual(pen.states[0].history, [1, 2, 3, 4])
        self.assertEqual(pen.seen_output_lens, [4])

        pen.cumulate_output_tokens(torch.tensor([4]))
        self.assertEqual(pen.states[0].history, [1, 2, 3, 4])

        req.output_ids.extend([5, 6, 7])
        pen.cumulate_output_tokens(torch.tensor([7]))
        self.assertEqual(pen.states[0].history, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(pen.seen_output_lens, [7])


# _BatchedPenalizer base class edge cases
class TestBatchedPenalizerBase(CustomTestCase):
    def test_filter_when_not_prepared_is_noop(self):
        """Test that filter on an unprepared penalizer does not crash."""
        reqs = [_make_req()]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        pen = orch.penalizers[BatchedFrequencyPenalizer]
        # pen is not prepared (frequency_penalty=0 → not required)
        pen.filter(torch.tensor([0]))  # should not raise

    def test_merge_prepares_both_if_needed(self):
        """Test that merge prepares unprepared side before concatenating."""
        reqs_a = [_make_req(freq=0.0)]  # not required
        reqs_b = [_make_req(freq=1.0)]  # required
        batch_a = _make_batch(reqs_a)
        batch_b = _make_batch(reqs_b)
        orch_a = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch_a, {BatchedFrequencyPenalizer}
        )
        orch_b = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch_b, {BatchedFrequencyPenalizer}
        )
        pen_a = orch_a.penalizers[BatchedFrequencyPenalizer]
        pen_b = orch_b.penalizers[BatchedFrequencyPenalizer]
        self.assertFalse(pen_a.is_prepared())
        self.assertTrue(pen_b.is_prepared())
        # Merge should prepare pen_a first
        pen_a.merge(pen_b)
        self.assertTrue(pen_a.is_prepared())
        self.assertEqual(pen_a.frequency_penalties.shape[0], 2)

    def test_merge_both_unprepared_is_noop(self):
        """Test that merging two unprepared penalizers keeps them unprepared."""
        reqs = [_make_req()]
        batch = _make_batch(reqs)
        orch1 = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        orch2 = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        pen1 = orch1.penalizers[BatchedFrequencyPenalizer]
        pen2 = orch2.penalizers[BatchedFrequencyPenalizer]
        pen1.merge(pen2)  # both not prepared → noop
        self.assertFalse(pen1.is_prepared())

    def test_prepare_is_idempotent(self):
        """Test that calling prepare() multiple times does not crash."""
        reqs = [_make_req(freq=1.0)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        pen = orch.penalizers[BatchedFrequencyPenalizer]
        self.assertTrue(pen.is_prepared())
        # Calling prepare again should not crash or reinitialize
        pen.prepare()
        self.assertTrue(pen.is_prepared())


# Orchestrator with multiple penalizer types
class TestOrchestratorMultiplePenalizers(CustomTestCase):
    def test_all_three_penalizers(self):
        """Test orchestrator managing frequency, presence, and min_new_tokens together."""
        reqs = [_make_req(freq=1.0, presence=0.5, min_tokens=2, eos_id=2)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE,
            batch,
            {
                BatchedFrequencyPenalizer,
                BatchedPresencePenalizer,
                BatchedMinNewTokensPenalizer,
            },
        )
        self.assertTrue(orch.is_required)

        # Cumulate one token
        output_ids = torch.tensor([5])
        orch.cumulate_output_tokens(output_ids)

        # Apply all penalties
        logits = torch.zeros(1, VOCAB_SIZE)
        orch.apply(logits)

        # Token 5: freq_penalty=1.0 (cumulated once) + pres_penalty=0.5
        self.assertAlmostEqual(logits[0, 5].item(), -1.5, places=4)
        # EOS (token 2): blocked by min_new_tokens (len=1 < min=2)
        self.assertTrue(torch.isinf(logits[0, 2]) and logits[0, 2] < 0)

    def test_filter_with_penalizer_no_longer_required(self):
        """Test that penalizer is torn down when no longer required after filter."""
        reqs = [_make_req(freq=0.0), _make_req(freq=1.0)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        self.assertTrue(orch.is_required)

        # Keep only the request with freq=0 (index 0)
        batch.reqs = [reqs[0]]
        orch.filter(torch.tensor([0]))

        pen = orch.penalizers[BatchedFrequencyPenalizer]
        # After filter, only req with freq=0 remains → penalizer not required
        self.assertFalse(pen.is_required())

    def test_filter_keeps_required_penalizer(self):
        """Test that filter keeps penalizer active when still required."""
        reqs = [_make_req(freq=1.0), _make_req(freq=2.0)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        self.assertTrue(orch.is_required)

        batch.reqs = [reqs[1]]
        orch.filter(torch.tensor([1]))
        self.assertTrue(orch.is_required)

    def test_merge_one_required(self):
        """Test that merge marks orchestrator as required when one side is."""
        reqs_a = [_make_req(freq=0.0)]
        reqs_b = [_make_req(freq=1.0)]
        batch_a = _make_batch(reqs_a)
        batch_b = _make_batch(reqs_b)
        orch_a = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch_a, {BatchedFrequencyPenalizer}
        )
        orch_b = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch_b, {BatchedFrequencyPenalizer}
        )
        self.assertFalse(orch_a.is_required)
        self.assertTrue(orch_b.is_required)

        orch_a.merge(orch_b)
        self.assertTrue(orch_a.is_required)
        pen = orch_a.penalizers[BatchedFrequencyPenalizer]
        self.assertEqual(pen.frequency_penalties.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
