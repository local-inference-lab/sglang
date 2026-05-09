import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Tuple

import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer

logger = logging.getLogger(__name__)

_MASK64 = (1 << 64) - 1
_HASH_BASE = 1_000_003
_LOG_INITIAL_APPLICATIONS = 8
_LOG_APPLICATION_INTERVAL = 128


def _sampling_param(sampling_params, name: str, default):
    values = getattr(sampling_params, "__dict__", None)
    if values is not None:
        value = values.get(name, default)
    else:
        value = getattr(sampling_params, name, default)
    return default if value is None else value


@dataclass(frozen=True)
class _EntropyPenaltyDetail:
    penalty: float
    match_len: int
    repeat_count: int


@dataclass
class _EntropyPenaltyState:
    penalty: float
    min_len: int
    max_len: int
    window: int
    max_penalty: float
    min_repetitions: int
    history: List[int] = field(default_factory=list)
    prefix_hashes: List[int] = field(default_factory=lambda: [0])
    index: DefaultDict[Tuple[int, int], Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    base_powers: List[int] = field(default_factory=list)
    indexed_match_lengths: List[int] = field(default_factory=list)
    logged_applications: int = 0

    @property
    def max_history_len(self) -> int:
        return self.window + self.max_len + 1

    @property
    def trim_stride(self) -> int:
        return max(1, self.max_len)

    @property
    def trim_threshold_len(self) -> int:
        return self.max_history_len + self.trim_stride

    def __post_init__(self):
        self.base_powers = [1]
        for _ in range(self.max_len):
            self.base_powers.append((self.base_powers[-1] * _HASH_BASE) & _MASK64)
        self.indexed_match_lengths = self._build_indexed_match_lengths()

    def append(self, token_id: int):
        if self.penalty == 0.0:
            return

        self.history.append(token_id)
        self.prefix_hashes.append(
            (self.prefix_hashes[-1] * _HASH_BASE + self._hash_token(token_id)) & _MASK64
        )

        if len(self.history) >= self.trim_threshold_len:
            self._trim_history(len(self.history) - self.max_history_len)

        self._index_continuation(len(self.history) - 1)

    def penalties(self) -> Dict[int, float]:
        return {
            token_id: detail.penalty
            for token_id, detail in self.penalty_details().items()
        }

    def penalty_details(self) -> Dict[int, _EntropyPenaltyDetail]:
        if self.penalty == 0.0 or len(self.history) < self.min_len:
            return {}

        details: Dict[int, _EntropyPenaltyDetail] = {}
        self._collect_lz_details(details)
        self._collect_periodic_details(details)
        return details

    def _collect_lz_details(self, details: Dict[int, _EntropyPenaltyDetail]) -> None:
        max_len = min(self.max_len, len(self.history))
        for match_len in self._iter_indexed_match_lengths(max_len):
            key = (
                match_len,
                self._span_hash(len(self.history) - match_len, len(self.history)),
            )
            continuations = self.index.get(key)
            if not continuations:
                continue

            for token_id, repeat_count in continuations.items():
                if repeat_count < self.min_repetitions:
                    continue

                penalty = self._score(match_len, repeat_count)
                detail = details.get(token_id)
                if detail is None or self._is_better_detail(
                    penalty, match_len, repeat_count, detail
                ):
                    details[token_id] = _EntropyPenaltyDetail(
                        penalty=penalty,
                        match_len=match_len,
                        repeat_count=repeat_count,
                    )

    def _collect_periodic_details(
        self, details: Dict[int, _EntropyPenaltyDetail]
    ) -> None:
        max_period = min(self.min_len - 1, self.max_len - 1, len(self.history) - 1)
        for period in range(1, max_period + 1):
            match_len = self._longest_periodic_suffix(period)
            if match_len < self.min_len:
                continue

            token_id = self.history[-period]
            repeat_count = max(1, match_len // period)
            if repeat_count < self.min_repetitions:
                continue

            penalty = self._score(match_len, repeat_count)
            detail = details.get(token_id)
            if detail is None or self._is_better_detail(
                penalty, match_len, repeat_count, detail
            ):
                details[token_id] = _EntropyPenaltyDetail(
                    penalty=penalty,
                    match_len=match_len,
                    repeat_count=repeat_count,
                )

    def _longest_periodic_suffix(self, period: int) -> int:
        max_copied_len = min(self.max_len, len(self.history) - period)
        if max_copied_len <= 0:
            return 0

        low = 0
        high = max_copied_len
        while low < high:
            mid = (low + high + 1) // 2
            if self._span_hash(len(self.history) - mid, len(self.history)) == (
                self._span_hash(
                    len(self.history) - period - mid,
                    len(self.history) - period,
                )
            ):
                low = mid
            else:
                high = mid - 1

        if low <= 0:
            return 0

        if period < self.min_len:
            aggregate_len = min(self.max_len, period + low)
            return aggregate_len if aggregate_len >= self.min_len else 0

        return low if low >= self.min_len else 0

    def _score(self, match_len: int, repeat_count: int) -> float:
        length_factor = match_len / self.min_len
        penalty = self.penalty * length_factor * repeat_count
        return min(penalty, self.max_penalty)

    @staticmethod
    def _is_better_detail(
        penalty: float,
        match_len: int,
        repeat_count: int,
        detail: _EntropyPenaltyDetail,
    ) -> bool:
        if penalty != detail.penalty:
            return penalty > detail.penalty
        if match_len != detail.match_len:
            return match_len > detail.match_len
        return repeat_count > detail.repeat_count

    def _rebuild(self):
        self._rebuild_prefix_hashes()
        self.index = defaultdict(Counter)
        for continuation_pos in range(1, len(self.history)):
            self._index_continuation(continuation_pos)

    def _build_indexed_match_lengths(self) -> List[int]:
        if self.max_len - self.min_len <= 64:
            return list(range(self.min_len, self.max_len + 1))

        range_size = self.max_len - self.min_len + 1
        stride = max(1, self.min_len, (range_size + 31) // 32)
        lengths = list(range(self.min_len, self.max_len + 1, stride))
        if lengths[-1] != self.max_len:
            lengths.append(self.max_len)
        return lengths

    def _iter_indexed_match_lengths(self, max_len: int):
        for match_len in self.indexed_match_lengths:
            if match_len > max_len:
                break
            yield match_len

    def _rebuild_prefix_hashes(self):
        self.prefix_hashes = [0]
        for token_id in self.history:
            self.prefix_hashes.append(
                (self.prefix_hashes[-1] * _HASH_BASE + self._hash_token(token_id))
                & _MASK64
            )

    def _trim_history(self, trim_len: int):
        if trim_len <= 0:
            return

        history_len = len(self.history)
        for start in range(trim_len):
            max_match_len = min(self.max_len, history_len - start - 1)
            for match_len in self._iter_indexed_match_lengths(max_match_len):
                continuation_pos = start + match_len
                token_id = self.history[continuation_pos]
                key = (match_len, self._span_hash(start, continuation_pos))
                continuations = self.index.get(key)
                if not continuations:
                    continue

                continuations[token_id] -= 1
                if continuations[token_id] <= 0:
                    del continuations[token_id]
                if not continuations:
                    del self.index[key]

        del self.history[:trim_len]
        self._rebuild_prefix_hashes()

    def _index_continuation(self, continuation_pos: int):
        max_len = min(self.max_len, continuation_pos)
        if max_len < self.min_len:
            return

        token_id = self.history[continuation_pos]
        for match_len in self._iter_indexed_match_lengths(max_len):
            start = continuation_pos - match_len
            key = (match_len, self._span_hash(start, continuation_pos))
            self.index[key][token_id] += 1

    def _span_hash(self, start: int, end: int) -> int:
        length = end - start
        return (
            self.prefix_hashes[end]
            - self.prefix_hashes[start] * self.base_powers[length]
        ) & _MASK64

    @staticmethod
    def _hash_token(token_id: int) -> int:
        return (int(token_id) + 1) & _MASK64


class BatchedEntropyPenalizer(_BatchedPenalizer):
    """
    LZ77-style self-compressibility penalizer.

    The penalizer tracks generated output tokens only. If a candidate next token
    would extend a long suffix that appeared earlier in the output stream, it
    applies a soft additive logit penalty proportional to match length and
    repeat count.
    """

    def _is_required(self) -> bool:
        return any(
            float(_sampling_param(req.sampling_params, "entropy_penalty", 0.0)) != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        reqs = self.orchestrator.reqs()
        self.request_ids: List[Any] = [getattr(req, "rid", None) for req in reqs]
        self.seen_output_lens: List[int] = [
            len(getattr(req, "output_ids", None) or []) for req in reqs
        ]
        self.states = [
            _EntropyPenaltyState(
                penalty=float(
                    _sampling_param(req.sampling_params, "entropy_penalty", 0.0)
                ),
                min_len=int(
                    _sampling_param(req.sampling_params, "entropy_penalty_min_len", 16)
                ),
                max_len=int(
                    _sampling_param(req.sampling_params, "entropy_penalty_max_len", 256)
                ),
                window=int(
                    _sampling_param(req.sampling_params, "entropy_penalty_window", 8192)
                ),
                max_penalty=float(
                    _sampling_param(
                        req.sampling_params, "entropy_penalty_max_penalty", 8.0
                    )
                ),
                min_repetitions=int(
                    _sampling_param(
                        req.sampling_params, "entropy_penalty_min_repetitions", 1
                    )
                ),
            )
            for req in reqs
        ]

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        output_ids_list = output_ids.tolist()
        for batch_idx, (state, req, token_id) in enumerate(
            zip(self.states, self.orchestrator.reqs(), output_ids_list)
        ):
            self._cumulate_req_output_tokens(
                batch_idx=batch_idx,
                state=state,
                req=req,
                token_id=int(token_id),
            )

    def _cumulate_req_output_tokens(
        self,
        batch_idx: int,
        state: _EntropyPenaltyState,
        req,
        token_id: int,
    ) -> None:
        output_history = getattr(req, "output_ids", None)
        if output_history is not None:
            seen_output_len = self.seen_output_lens[batch_idx]
            if len(output_history) > seen_output_len:
                for new_token_id in output_history[seen_output_len:]:
                    state.append(int(new_token_id))
                self.seen_output_lens[batch_idx] = len(output_history)
                return

            # Overlap mode can feed the last prompt token before any output exists.
            origin_input_ids = getattr(req, "origin_input_ids", None)
            if output_history == [] and origin_input_ids:
                if token_id == int(origin_input_ids[-1]):
                    return

            # When output_history is non-empty and already consumed, it is more
            # authoritative than the delayed one-token tensor. Avoid double-counting.
            if output_history:
                return

        state.append(token_id)
        if output_history is not None:
            self.seen_output_lens[batch_idx] += 1

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.shape[1]
        for batch_idx, state in enumerate(self.states):
            penalty_details = {
                token_id: detail
                for token_id, detail in state.penalty_details().items()
                if 0 <= token_id < vocab_size and detail.penalty != 0.0
            }
            if not penalty_details:
                continue

            state.logged_applications += 1
            if (
                state.logged_applications <= _LOG_INITIAL_APPLICATIONS
                or state.logged_applications % _LOG_APPLICATION_INTERVAL == 0
            ):
                best_token_id, best_detail = max(
                    penalty_details.items(),
                    key=lambda item: (
                        item[1].penalty,
                        item[1].match_len,
                        item[1].repeat_count,
                    ),
                )
                logger.info(
                    "entropy_penalty applied request_id=%s history_tokens=%d "
                    "candidates=%d top_token=%d penalty=%.3f match_len=%d "
                    "repeat_count=%d application_count=%d",
                    self.request_ids[batch_idx]
                    if batch_idx < len(self.request_ids)
                    else None,
                    len(state.history),
                    len(penalty_details),
                    best_token_id,
                    best_detail.penalty,
                    best_detail.match_len,
                    best_detail.repeat_count,
                    state.logged_applications,
                )

            token_ids = torch.tensor(
                list(penalty_details.keys()), dtype=torch.long, device=logits.device
            )
            penalty_values = torch.tensor(
                [detail.penalty for detail in penalty_details.values()],
                dtype=logits.dtype,
                device=logits.device,
            )
            logits[batch_idx, token_ids] -= penalty_values
        return logits

    def _filter(self, keep_indices: torch.Tensor):
        keep_indices_list = keep_indices.tolist()
        self.states = [self.states[i] for i in keep_indices_list]
        self.request_ids = [self.request_ids[i] for i in keep_indices_list]
        self.seen_output_lens = [self.seen_output_lens[i] for i in keep_indices_list]

    def _merge(self, their: "BatchedEntropyPenalizer"):
        self.states.extend(their.states)
        self.request_ids.extend(their.request_ids)
        self.seen_output_lens.extend(their.seen_output_lens)

    def _teardown(self) -> None:
        if hasattr(self, "states"):
            delattr(self, "states")
        if hasattr(self, "request_ids"):
            delattr(self, "request_ids")
        if hasattr(self, "seen_output_lens"):
            delattr(self, "seen_output_lens")
