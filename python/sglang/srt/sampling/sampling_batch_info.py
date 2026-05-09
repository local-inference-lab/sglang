from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

import sglang.srt.sampling.penaltylib as penaltylib
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


logger = logging.getLogger(__name__)

_THINKING_END_LOGIT_BOOST_LOG_INITIAL_APPLICATIONS = 8
_THINKING_END_LOGIT_BOOST_LOG_APPLICATION_INTERVAL = 128


@dataclasses.dataclass
class SamplingBatchInfo:
    # Basic batched sampling params
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor

    # Whether all requests use greedy sampling
    is_all_greedy: bool

    # Whether any requests use top_p sampling
    need_top_p_sampling: bool

    # Whether any requests use top_k sampling
    need_top_k_sampling: bool

    # Whether any request needs min_p sampling
    need_min_p_sampling: bool

    # Masking tensors for grammar-guided structured outputs
    vocab_size: int
    grammars: Optional[List] = None
    vocab_mask: Optional[torch.Tensor] = None
    apply_mask_func: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None

    # Penalizer
    penalizer_orchestrator: Optional[penaltylib.BatchedPenalizerOrchestrator] = None
    acc_additive_penalties: Optional[torch.Tensor] = None  # Used in the overlap mode
    acc_scaling_penalties: Optional[torch.Tensor] = (
        None  # Used in the overlap mode for repetition penalty
    )

    # Whether any request has custom logit processor
    has_custom_logit_processor: bool = False
    # Custom parameters
    custom_params: Optional[List[Optional[Dict[str, Any]]]] = None
    # Custom logit processor
    custom_logit_processor: Optional[
        Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]
    ] = None

    # Used for deterministic sampling
    sampling_seed: Optional[torch.Tensor] = None

    # Device
    device: str = "cuda"

    # Handle logit bias
    logit_bias: Optional[torch.Tensor] = None

    # Slowly bias an open thinking block toward its end token.
    has_thinking_end_logit_boost: bool = False
    thinking_start_token_ids: Optional[List[Optional[int]]] = None
    thinking_end_token_ids: Optional[List[Optional[int]]] = None
    thinking_end_logit_boosts: Optional[List[float]] = None
    thinking_end_logit_boost_start_tokens: Optional[List[int]] = None
    thinking_end_logit_boost_ramp_tokens: Optional[List[int]] = None
    thinking_end_logit_boost_reqs: Optional[List[Any]] = None
    thinking_end_logit_boost_logged_applications: Optional[List[int]] = None

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
        global_server_args = get_global_server_args()
        enable_deterministic = global_server_args.enable_deterministic_inference

        reqs = batch.reqs
        device = batch.device
        temperatures = torch.tensor(
            [r.sampling_params.temperature for r in reqs],
            dtype=torch.float,
            device=device,
        ).view(-1, 1)
        top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs], dtype=torch.float, device=device
        )
        top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs], dtype=torch.int32, device=device
        )
        min_ps = torch.tensor(
            [r.sampling_params.min_p for r in reqs], dtype=torch.float, device=device
        )
        sampling_seed = (
            torch.tensor(
                [
                    (
                        r.sampling_params.sampling_seed
                        if r.sampling_params.sampling_seed is not None
                        else 42
                    )
                    for r in reqs
                ],
                dtype=torch.int64,
                device=device,
            )
            if enable_deterministic
            else None
        )

        logit_bias = None
        if any(r.sampling_params.logit_bias is not None for r in reqs):
            logit_bias = torch.zeros(len(reqs), vocab_size, device=device)
            for i, r in enumerate(reqs):
                if r.sampling_params.logit_bias is not None:
                    for key, value in r.sampling_params.logit_bias.items():
                        logit_bias[i, int(key)] = value

        thinking_end_logit_boosts = [
            float(r.sampling_params.thinking_end_logit_boost) for r in reqs
        ]
        has_thinking_end_logit_boost = any(
            boost > 0.0 for boost in thinking_end_logit_boosts
        )
        if has_thinking_end_logit_boost:
            thinking_start_token_ids = [
                r.sampling_params.thinking_start_token_id for r in reqs
            ]
            thinking_end_token_ids = [
                r.sampling_params.thinking_end_token_id for r in reqs
            ]
            thinking_end_logit_boost_start_tokens = [
                int(r.sampling_params.thinking_end_logit_boost_start) for r in reqs
            ]
            thinking_end_logit_boost_ramp_tokens = [
                int(r.sampling_params.thinking_end_logit_boost_ramp) for r in reqs
            ]
            thinking_end_logit_boost_reqs = list(reqs)
            thinking_end_logit_boost_logged_applications = [0] * len(reqs)
        else:
            thinking_start_token_ids = None
            thinking_end_token_ids = None
            thinking_end_logit_boosts = None
            thinking_end_logit_boost_start_tokens = None
            thinking_end_logit_boost_ramp_tokens = None
            thinking_end_logit_boost_reqs = None
            thinking_end_logit_boost_logged_applications = None

        # Check if any request has custom logit processor
        has_custom_logit_processor = (
            global_server_args.enable_custom_logit_processor
            and any(r.custom_logit_processor for r in reqs)  # check the flag first.
        )  # then check the requests.

        if has_custom_logit_processor:
            # Merge the same type of custom logit processors together
            processor_dict = {}
            for i, r in enumerate(reqs):
                if r.custom_logit_processor is None:
                    continue
                processor_str = r.custom_logit_processor
                if processor_str not in processor_dict:
                    processor_dict[processor_str] = []
                processor_dict[processor_str].append(i)

            merged_custom_logit_processor = {
                hash(processor_str): (
                    # The deserialized custom logit processor object
                    CustomLogitProcessor.from_str(processor_str),
                    # The mask tensor for the requests that use this custom logit processor
                    torch.zeros(len(reqs), dtype=torch.bool)
                    .scatter_(0, torch.tensor(true_indices), True)
                    .to(device, non_blocking=True),
                )
                for processor_str, true_indices in processor_dict.items()
            }
            custom_params = [r.sampling_params.custom_params for r in reqs]
        else:
            merged_custom_logit_processor = None
            custom_params = None

        # Each penalizers will do nothing if they evaluate themselves as not required by looking at
        # the sampling_params of the requests (See {_is_required()} of each penalizers). So this
        # should not add hefty computation overhead other than simple checks.
        #
        # While we can choose not to even create the class instances if they are not required, this
        # could add additional complexity to the {ScheduleBatch} class, especially we need to
        # handle {filter_batch()} and {merge_batch()} cases as well.
        penalizer_orchestrator = penaltylib.BatchedPenalizerOrchestrator(
            vocab_size=vocab_size,
            batch=batch,
            penalizers={
                penaltylib.BatchedEntropyPenalizer,
                penaltylib.BatchedFrequencyPenalizer,
                penaltylib.BatchedMinNewTokensPenalizer,
                penaltylib.BatchedPresencePenalizer,
                penaltylib.BatchedRepetitionPenalizer,
            },
        )

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            sampling_seed=sampling_seed,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            need_top_p_sampling=any(r.sampling_params.top_p != 1.0 for r in reqs),
            need_top_k_sampling=any(r.sampling_params.top_k != TOP_K_ALL for r in reqs),
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
            vocab_size=vocab_size,
            penalizer_orchestrator=penalizer_orchestrator,
            has_custom_logit_processor=has_custom_logit_processor,
            custom_params=custom_params,
            custom_logit_processor=merged_custom_logit_processor,
            device=device,
            logit_bias=logit_bias,
            has_thinking_end_logit_boost=has_thinking_end_logit_boost,
            thinking_start_token_ids=thinking_start_token_ids,
            thinking_end_token_ids=thinking_end_token_ids,
            thinking_end_logit_boosts=thinking_end_logit_boosts,
            thinking_end_logit_boost_start_tokens=thinking_end_logit_boost_start_tokens,
            thinking_end_logit_boost_ramp_tokens=thinking_end_logit_boost_ramp_tokens,
            thinking_end_logit_boost_reqs=thinking_end_logit_boost_reqs,
            thinking_end_logit_boost_logged_applications=(
                thinking_end_logit_boost_logged_applications
            ),
        )
        ret.adjusted_from_schedule_batch(batch, vocab_size)
        return ret

    # placeholder for override
    def adjusted_from_schedule_batch(self, batch: ScheduleBatch, vocab_size: int):
        pass

    # placeholder for override
    def adjusted_merge_batch(self, other: "SamplingBatchInfo"):
        pass

    # placeholder for override
    def adjusted_filter_batch(
        self, keep_indices: List[int], keep_indices_device: torch.Tensor
    ):
        pass

    def __len__(self):
        return len(self.temperatures)

    def update_regex_vocab_mask(self):
        if not self.grammars:
            self.vocab_mask = None
            self.apply_mask_func = None
            return

        # Find a grammar from the list
        first_grammar = next(grammar for grammar in self.grammars if grammar)

        # TODO(lianmin): Maybe we can reuse the existing mask?
        self.vocab_mask = first_grammar.allocate_vocab_mask(
            vocab_size=self.vocab_size,
            batch_size=len(self.temperatures),
            device=self.device,
        )
        self.apply_mask_func = (
            first_grammar.apply_vocab_mask
        )  # force to use static method

        # Apply the mask
        for i, grammar in enumerate(self.grammars):
            if grammar and not grammar.finished and not grammar.is_terminated():
                grammar.fill_vocab_mask(self.vocab_mask, i)

        # Move the mask to the device if needed
        self.vocab_mask = first_grammar.move_vocab_mask(self.vocab_mask, self.device)

    def update_penalties(self):
        if self.penalizer_orchestrator.is_required:
            self.acc_additive_penalties = torch.zeros(
                (len(self.temperatures), self.vocab_size),
                dtype=torch.float32,
                device=self.temperatures.device,
            )
            self.penalizer_orchestrator.accumulate_additive_penalties(
                self.acc_additive_penalties
            )
            self.acc_scaling_penalties = (
                self.penalizer_orchestrator.accumulate_scaling_penalties()
            )
        else:
            self.acc_additive_penalties = None
            self.acc_scaling_penalties = None

    def apply_logits_bias(self, logits: torch.Tensor):
        if self.acc_additive_penalties is not None:
            # Used in the overlap mode
            logits.add_(self.acc_additive_penalties)

        if self.acc_scaling_penalties is not None:
            # Used in the overlap mode
            apply_scaling_penalties(logits, self.acc_scaling_penalties)

        if self.penalizer_orchestrator and self.penalizer_orchestrator.is_required:
            # Used in the non-overlap mode
            self.penalizer_orchestrator.apply(logits)

        if self.vocab_mask is not None:
            self.apply_mask_func(logits=logits, vocab_mask=self.vocab_mask)

        if self.logit_bias is not None:
            logits.add_(self.logit_bias)

        if self.has_thinking_end_logit_boost:
            self.apply_thinking_end_logit_boost(logits)

    def apply_thinking_end_logit_boost(
        self, logits: torch.Tensor, repeat: int = 1
    ) -> None:
        if not self.has_thinking_end_logit_boost:
            return

        if repeat < 1:
            raise ValueError(f"repeat must be at least 1, got {repeat}.")
        assert logits.shape[0] == len(self) * repeat, (
            f"The batch size of logits ({logits.shape[0]}) does not match the batch "
            f"size of sampling info ({len(self)}) x repeat ({repeat})."
        )

        row_indices: List[int] = []
        token_ids: List[int] = []
        boost_values: List[float] = []

        assert self.thinking_end_logit_boost_reqs is not None
        assert self.thinking_end_logit_boosts is not None
        assert self.thinking_start_token_ids is not None
        assert self.thinking_end_token_ids is not None
        assert self.thinking_end_logit_boost_start_tokens is not None
        assert self.thinking_end_logit_boost_ramp_tokens is not None
        if self.thinking_end_logit_boost_logged_applications is None:
            self.thinking_end_logit_boost_logged_applications = [0] * len(self)

        for i, req in enumerate(self.thinking_end_logit_boost_reqs):
            boost = self.thinking_end_logit_boosts[i]
            if boost <= 0.0 or req is None:
                continue

            start_token_id = self.thinking_start_token_ids[i]
            end_token_id = self.thinking_end_token_ids[i]
            if start_token_id is None or end_token_id is None:
                continue

            cur_ids = [*req.origin_input_ids, *req.output_ids]
            start_index = self._last_token_index(cur_ids, start_token_id)
            if start_index is None:
                continue

            end_index = self._last_token_index(cur_ids, end_token_id)
            if end_index is not None and end_index > start_index:
                continue

            depth = len(cur_ids) - start_index - 1
            depth_after_cutin = depth - self.thinking_end_logit_boost_start_tokens[i]
            if depth_after_cutin < 0:
                continue

            ramp_tokens = self.thinking_end_logit_boost_ramp_tokens[i]
            ramp_fraction = (
                1.0 if ramp_tokens == 0 else min(depth_after_cutin / ramp_tokens, 1.0)
            )
            boost_value = boost * ramp_fraction
            if boost_value == 0.0 or not 0 <= end_token_id < logits.shape[-1]:
                continue

            self.thinking_end_logit_boost_logged_applications[i] += 1
            application_count = self.thinking_end_logit_boost_logged_applications[i]
            if (
                application_count
                <= _THINKING_END_LOGIT_BOOST_LOG_INITIAL_APPLICATIONS
                or application_count
                % _THINKING_END_LOGIT_BOOST_LOG_APPLICATION_INTERVAL
                == 0
            ):
                logger.info(
                    "thinking_end_logit_boost applied request_id=%s depth=%d "
                    "end_token=%d boost=%.3f max_boost=%.3f ramp_fraction=%.3f "
                    "cutin=%d ramp=%d repeat=%d application_count=%d",
                    getattr(req, "rid", None),
                    depth,
                    end_token_id,
                    boost_value,
                    boost,
                    ramp_fraction,
                    self.thinking_end_logit_boost_start_tokens[i],
                    ramp_tokens,
                    repeat,
                    application_count,
                )

            row_indices.append(i)
            token_ids.append(end_token_id)
            boost_values.append(boost_value)

        if not row_indices:
            return

        rows = torch.tensor(row_indices, device=logits.device, dtype=torch.long)
        cols = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
        values = torch.tensor(boost_values, device=logits.device, dtype=logits.dtype)
        if repeat != 1:
            offsets = torch.arange(repeat, device=logits.device, dtype=torch.long)
            rows = (rows * repeat).repeat_interleave(repeat) + offsets.repeat(
                len(row_indices)
            )
            cols = cols.repeat_interleave(repeat)
            values = values.repeat_interleave(repeat)

        logits[rows, cols] += values

    @staticmethod
    def _last_token_index(token_ids: List[int], token_id: int) -> Optional[int]:
        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] == token_id:
                return i
        return None

    def filter_batch(self, keep_indices: List[int], keep_indices_device: torch.Tensor):
        self.penalizer_orchestrator.filter(keep_indices_device)

        if self.has_custom_logit_processor:
            self._filter_batch_custom_logit_processor(keep_indices, keep_indices_device)

        if self.has_thinking_end_logit_boost:
            self._filter_batch_thinking_end_logit_boost(keep_indices)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "sampling_seed",
        ]:
            value = getattr(self, item, None)
            if value is not None:
                setattr(self, item, value[keep_indices_device])

        if self.logit_bias is not None:
            self.logit_bias = self.logit_bias[keep_indices_device]

        self.adjusted_filter_batch(keep_indices, keep_indices_device)

    def _filter_batch_custom_logit_processor(
        self, keep_indices: List[int], keep_indices_device: torch.Tensor
    ):
        """Filter the custom logit processor and custom params"""
        self.custom_logit_processor = {
            k: (p, mask[keep_indices_device])
            for k, (p, mask) in self.custom_logit_processor.items()
            if torch.any(
                mask[keep_indices_device]
            )  # ignore the custom logit processor whose mask is all False
        }
        self.custom_params = [self.custom_params[i] for i in keep_indices]

        # If the custom logit processor is an empty dict, set the flag to False,
        # and set the custom logit processor and custom params to None.
        if len(self.custom_logit_processor) == 0:
            self.custom_logit_processor = None
            self.custom_params = None
            self.has_custom_logit_processor = False

    def _filter_batch_thinking_end_logit_boost(self, keep_indices: List[int]):
        for item in [
            "thinking_start_token_ids",
            "thinking_end_token_ids",
            "thinking_end_logit_boosts",
            "thinking_end_logit_boost_start_tokens",
            "thinking_end_logit_boost_ramp_tokens",
            "thinking_end_logit_boost_reqs",
            "thinking_end_logit_boost_logged_applications",
        ]:
            values = getattr(self, item)
            if values is None and item == "thinking_end_logit_boost_logged_applications":
                values = [0] * len(self)
            setattr(self, item, [values[i] for i in keep_indices])

        self._cleanup_thinking_end_logit_boost()

    @staticmethod
    def merge_custom_logit_processor(
        lhs: Optional[Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]],
        rhs: Optional[Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]],
        bs1: int,
        bs2: int,
        device: str,
    ):
        if lhs is None and rhs is None:
            return None
        lhs, rhs = lhs or {}, rhs or {}

        keys = set(lhs.keys()).union(set(rhs.keys()))
        merged_dict = {}

        for k in keys:
            # Get the logit processor object
            processor = lhs[k][0] if k in lhs else rhs[k][0]
            # Get and merge the mask tensors from the two dicts
            left_mask = (
                lhs[k][1]
                if k in lhs
                else torch.zeros(bs1, dtype=torch.bool, device=device)
            )
            right_mask = (
                rhs[k][1]
                if k in rhs
                else torch.zeros(bs2, dtype=torch.bool, device=device)
            )
            merged_dict[k] = (processor, torch.cat([left_mask, right_mask]))

            assert merged_dict[k][1].shape[0] == bs1 + bs2, (
                f"The batch size of merged mask ({merged_dict[k][1].shape[0]}) does not match "
                f"the sum of the batch sizes of the two masks ({bs1 + bs2})"
                f"\n{left_mask=}\n{right_mask=}\n{bs1=}\n{bs2=}"
                f"\n{lhs=}\n{rhs=}"
            )

        return merged_dict

    def merge_batch(self, other: "SamplingBatchInfo"):
        self.penalizer_orchestrator.merge(other.penalizer_orchestrator)

        # Merge the custom logit processors and custom params lists
        if self.has_custom_logit_processor or other.has_custom_logit_processor:
            # Merge the custom logit processors
            self.custom_logit_processor = (
                SamplingBatchInfo.merge_custom_logit_processor(
                    self.custom_logit_processor,
                    other.custom_logit_processor,
                    len(self),
                    len(other),
                    self.device,
                )
            )
            # Merge the custom params lists
            self.custom_params = self.custom_params or [None] * len(self)
            other.custom_params = other.custom_params or [None] * len(other)
            self.custom_params.extend(other.custom_params)

            # Set the flag to True if any of the two has custom logit processor
            self.has_custom_logit_processor = True

        # Merge logit bias - note this has to come before the temperatures tensor update! Otherwise will cause crashes.
        # See note below on len(self) and len(other).
        self.logit_bias = merge_bias_tensor(
            self.logit_bias, other.logit_bias, len(self), len(other), self.device, 0.0
        )

        if self.has_thinking_end_logit_boost or other.has_thinking_end_logit_boost:
            self._merge_thinking_end_logit_boost(other)

        # Note: because the __len()__ operator is defined on the temperatures tensor,
        # please make sure any merge operation with len(self) or len(other) is done before
        # the merge operation of the temperatures tensor below.
        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "sampling_seed",
        ]:
            self_val = getattr(self, item, None)
            other_val = getattr(other, item, None)
            if self_val is not None and other_val is not None:
                setattr(self, item, torch.cat([self_val, other_val]))

        self.is_all_greedy &= other.is_all_greedy
        self.need_top_p_sampling |= other.need_top_p_sampling
        self.need_top_k_sampling |= other.need_top_k_sampling
        self.need_min_p_sampling |= other.need_min_p_sampling

        self.adjusted_merge_batch(other)

    def _merge_thinking_end_logit_boost(self, other: "SamplingBatchInfo"):
        self._ensure_thinking_end_logit_boost_lists()
        other_values = other._thinking_end_logit_boost_lists_or_defaults()
        for item, values in other_values.items():
            getattr(self, item).extend(values)
        self.has_thinking_end_logit_boost = True

    def _ensure_thinking_end_logit_boost_lists(self):
        if self.has_thinking_end_logit_boost:
            return
        self.thinking_start_token_ids = [None] * len(self)
        self.thinking_end_token_ids = [None] * len(self)
        self.thinking_end_logit_boosts = [0.0] * len(self)
        self.thinking_end_logit_boost_start_tokens = [0] * len(self)
        self.thinking_end_logit_boost_ramp_tokens = [0] * len(self)
        self.thinking_end_logit_boost_reqs = [None] * len(self)
        self.thinking_end_logit_boost_logged_applications = [0] * len(self)
        self.has_thinking_end_logit_boost = True

    def _thinking_end_logit_boost_lists_or_defaults(self) -> Dict[str, List[Any]]:
        if self.has_thinking_end_logit_boost:
            return {
                "thinking_start_token_ids": list(self.thinking_start_token_ids),
                "thinking_end_token_ids": list(self.thinking_end_token_ids),
                "thinking_end_logit_boosts": list(self.thinking_end_logit_boosts),
                "thinking_end_logit_boost_start_tokens": list(
                    self.thinking_end_logit_boost_start_tokens
                ),
                "thinking_end_logit_boost_ramp_tokens": list(
                    self.thinking_end_logit_boost_ramp_tokens
                ),
                "thinking_end_logit_boost_reqs": list(
                    self.thinking_end_logit_boost_reqs
                ),
                "thinking_end_logit_boost_logged_applications": list(
                    self.thinking_end_logit_boost_logged_applications
                    or [0] * len(self)
                ),
            }
        return {
            "thinking_start_token_ids": [None] * len(self),
            "thinking_end_token_ids": [None] * len(self),
            "thinking_end_logit_boosts": [0.0] * len(self),
            "thinking_end_logit_boost_start_tokens": [0] * len(self),
            "thinking_end_logit_boost_ramp_tokens": [0] * len(self),
            "thinking_end_logit_boost_reqs": [None] * len(self),
            "thinking_end_logit_boost_logged_applications": [0] * len(self),
        }

    def _cleanup_thinking_end_logit_boost(self):
        if any(boost > 0.0 for boost in self.thinking_end_logit_boosts):
            return
        self.has_thinking_end_logit_boost = False
        self.thinking_start_token_ids = None
        self.thinking_end_token_ids = None
        self.thinking_end_logit_boosts = None
        self.thinking_end_logit_boost_start_tokens = None
        self.thinking_end_logit_boost_ramp_tokens = None
        self.thinking_end_logit_boost_reqs = None
        self.thinking_end_logit_boost_logged_applications = None

    def copy_for_forward(self):
        # Accumulate the penalty into a pre-allocated buffer to get rid of the dependency of `penalizer_orchestrator` later
        self.update_penalties()
        return dataclasses.replace(self, penalizer_orchestrator=None)


def merge_bias_tensor(
    lhs: Optional[torch.Tensor],
    rhs: Optional[torch.Tensor],
    bs1: int,
    bs2: int,
    device: str,
    default: float,
):
    """Merge two bias tensors for batch merging.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor
        bs1: Batch size of left-hand side tensor
        bs2: Batch size of right-hand side tensor
        device: Device to place the merged tensor on
        default: Default value for missing tensor elements

    Returns:
        Merged tensor or None if both inputs are None
    """
    if lhs is None and rhs is None:
        return None

    if lhs is not None and rhs is not None:
        return torch.cat([lhs, rhs])
    else:
        if lhs is not None:
            shape, dtype = lhs.shape[1:], lhs.dtype
        else:
            shape, dtype = rhs.shape[1:], rhs.dtype

        if lhs is None:
            lhs = torch.empty((bs1, *shape), device=device, dtype=dtype).fill_(default)
        if rhs is None:
            rhs = torch.empty((bs2, *shape), device=device, dtype=dtype).fill_(default)
        return torch.cat([lhs, rhs])
