# Copyright (c) 2026 AllSpark Research.
# This source code is licensed under the Apache 2.0 License.
#
# Append-only conversation log + context-management markers.
"""Single owner of the append-only conversation log and its context-management markers.

Why this exists
---------------
The eval harness threads ONE ``message_history`` list that is simultaneously (a) the
model-visible conversation and (b) — snapshotted each turn via ``TaskLog.save()`` — the
persisted trajectory. Context management mutates that list *in place*:

* ``discard_all_tool_history`` — reset to just the first user message,
* ``enforce_context_budget`` / ``drop_last_round`` — pop whole trailing rounds,
* ``ensure_summary_context`` — pop the last tool round before summarizing,
* assorted rollbacks (format-error / refusal / duplicate-query / tool-error / no-boxed).

Because the same object is both mutated and saved, any popped/discarded round vanishes from
the on-disk trace (log-fidelity loss), and the SFT exporter / per-turn replay cannot
reconstruct the exact *structural* conversation the model saw after a context op
(state-faithful-SFT loss).

Design (additive, equivalence-proven)
-------------------------------------
``ConversationHistory`` IS the model-visible list (its own elements are exactly what the
old plain list held, produced by the unchanged inference path). It *additionally* maintains
an append-only ``full`` log that records every structural op as a marker. Markers live ONLY
in ``full`` — never in the visible list, never sent to the model, never in the persisted
``message_history`` field — so the model and every existing reader are unaffected.

``replay(full) -> visible`` is the single-owner reconstruction (mirrors the marker replay in
AxisAgentic, https://github.com/XYZ-AI-Lab/AxisAgentic). By construction, after every
operation ``replay(self.full) == list(self)``; :meth:`ConversationHistory.assert_consistent`
checks it (used by tests and an optional runtime guard).

Marker wire-format (a plain message dict, role == :data:`MARKER_ROLE`)::

    {"role": "context_marker", "op": "rollback",     "count": N,       "reason": "..."}
    {"role": "context_marker", "op": "discard_all",  "prefix_len": K,  "reason": "..."}

A marker whose range does not fit the current visible list is treated as a no-op (a stale or
corrupt marker in a loaded trace), matching AxisAgentic's staleness-safe splices.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

MARKER_ROLE = "context_marker"
OP_ROLLBACK = "rollback"
OP_DISCARD_ALL = "discard_all"

__all__ = [
    "MARKER_ROLE",
    "OP_ROLLBACK",
    "OP_DISCARD_ALL",
    "build_rollback_marker",
    "build_discard_all_marker",
    "is_marker",
    "replay",
    "strip_markers",
    "ConversationHistory",
]


# --------------------------------------------------------------------------- #
# Marker builders / predicates (single owner of the wire-format)
# --------------------------------------------------------------------------- #
def build_rollback_marker(count: int, reason: Optional[str] = None) -> Dict[str, Any]:
    """Record that the last ``count`` visible messages were hidden (rolled back)."""
    marker: Dict[str, Any] = {"role": MARKER_ROLE, "op": OP_ROLLBACK, "count": int(count)}
    if reason:
        marker["reason"] = reason
    return marker


def build_discard_all_marker(prefix_len: int, reason: Optional[str] = None) -> Dict[str, Any]:
    """Record that the visible conversation was truncated to its leading ``prefix_len``."""
    marker: Dict[str, Any] = {"role": MARKER_ROLE, "op": OP_DISCARD_ALL, "prefix_len": int(prefix_len)}
    if reason:
        marker["reason"] = reason
    return marker


def is_marker(message: Any) -> bool:
    """True iff ``message`` is a context-management marker (not a real conversation turn)."""
    return isinstance(message, dict) and message.get("role") == MARKER_ROLE


def _as_int(value: Any) -> Optional[int]:
    # bool is an int subclass; reject it so a corrupted payload cannot splice.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


# --------------------------------------------------------------------------- #
# Replay: append-only full log -> model-visible conversation (single owner)
# --------------------------------------------------------------------------- #
def replay(full: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reconstruct the model-visible conversation by replaying ``full`` in order.

    Real messages append to the visible list; markers transform it:

    * ``rollback{count}``      -> hide the last ``count`` visible messages,
    * ``discard_all{prefix_len}`` -> truncate the visible list to its leading ``prefix_len``.

    Any marker whose recorded range does not fit the current visible list is a no-op
    (stale/malformed). Unknown ops are ignored. The returned list references the same
    message dicts as ``full`` (no copies), so it round-trips byte-for-byte.
    """
    visible: List[Dict[str, Any]] = []
    for message in full:
        if is_marker(message):
            op = message.get("op")
            if op == OP_ROLLBACK:
                count = _as_int(message.get("count"))
                # Staleness-safe (consistent with discard_all below): a rollback whose
                # count does not fit the current visible list is a no-op. Our tracked list
                # only ever emits count==1 for a real tail pop, so this always fits for
                # harness-produced logs; the guard only shields corrupt/stale markers.
                if count is not None and 0 < count <= len(visible):
                    del visible[len(visible) - count:]
            elif op == OP_DISCARD_ALL:
                prefix_len = _as_int(message.get("prefix_len"))
                if prefix_len is not None and 0 <= prefix_len <= len(visible):
                    visible = visible[:prefix_len]
            # unknown / malformed op -> ignore (no-op)
            continue
        visible.append(message)
    return visible


def strip_markers(full: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return ``full`` with marker messages removed (real messages only, order preserved).

    NOTE: this is *not* the model-view — it keeps rounds that were later rolled back /
    discarded. Use :func:`replay` to get what the model actually saw.
    """
    return [m for m in full if not is_marker(m)]


# --------------------------------------------------------------------------- #
# ConversationHistory: the tracked visible list that emits the append-only log
# --------------------------------------------------------------------------- #
_RUNTIME_CHECK = os.environ.get("MIROFLOW_CONV_LOG_CHECK", "0") == "1"


class ConversationHistory(list):
    """A ``list`` that is the model-visible conversation while maintaining an append-only
    ``full`` log (real messages + markers) for state-faithful logging / SFT / replay.

    Structural ops are intercepted:

    * ``append`` / ``extend`` / ``+=`` -> also appended to ``full``,
    * ``pop`` (tail only) -> records a ``rollback(1)`` marker in ``full``,
    * :meth:`discard_all_to_first_user` -> records a ``discard_all`` marker (the one
      reference-rebind site in the harness, converted to an in-place reset).

    ``copy()`` deliberately returns a plain, UNTRACKED ``list`` — the harness uses
    ``message_history.copy()`` for throwaway summary scratch, whose pops must NOT be
    recorded. Non-tail / bulk mutations (``insert`` / ``remove`` / ``clear`` /
    ``__setitem__`` / ``__delitem__`` / ``sort`` / ``reverse``) are blocked: they never
    occur on the stored list today (grep-verified) and would silently desync replay.

    INVARIANT: ``replay(self.full) == list(self)`` after every operation.
    """

    def __init__(self, iterable: Optional[Any] = None) -> None:
        seed = list(iterable) if iterable is not None else []
        super().__init__(seed)
        # Real seed messages (usually the single initial user/task turn). The full log is a
        # separate list object so slicing/truncating the visible view never touches it.
        self._full: List[Dict[str, Any]] = list(seed)
        self._pending_reason: Optional[str] = None

    # -- append-only log accessor ------------------------------------------- #
    @property
    def full(self) -> List[Dict[str, Any]]:
        """The append-only log: real messages interleaved with markers, in operation order."""
        return self._full

    # -- optional metadata: label the NEXT structural op (consumed once) ----- #
    def set_op_reason(self, reason: Optional[str]) -> None:
        self._pending_reason = reason

    def _consume_reason(self) -> Optional[str]:
        reason, self._pending_reason = self._pending_reason, None
        return reason

    # -- tracked mutators ---------------------------------------------------- #
    def append(self, item: Dict[str, Any]) -> None:
        super().append(item)
        self._full.append(item)
        self._maybe_check()

    def extend(self, items: Any) -> None:
        items = list(items)
        super().extend(items)
        self._full.extend(items)
        self._maybe_check()

    def __iadd__(self, other: Any) -> "ConversationHistory":
        self.extend(other)
        return self

    def pop(self, index: int = -1) -> Dict[str, Any]:
        n = len(self)
        norm = index if index >= 0 else n + index
        if norm != n - 1:
            raise RuntimeError(
                f"ConversationHistory.pop({index!r}) is not a tail pop (len={n}); "
                "append-only replay only supports tail pops."
            )
        item = super().pop(index)
        self._full.append(build_rollback_marker(1, reason=self._consume_reason()))
        self._maybe_check()
        return item

    def discard_all_to_first_user(self, reason: Optional[str] = OP_DISCARD_ALL) -> int:
        """Reset the visible conversation to only its first user message (HCM discard-all),
        recording a ``discard_all`` marker instead of rebinding to a fresh list so this object
        (and its ``full``) stays the canonical tracked history.

        Mirrors ``base_client.discard_all_tool_history`` (keep a deepcopy of the first user
        message). Returns the number of messages discarded (0 if no user message / nothing to
        discard).
        """
        first_user_idx = next(
            (i for i, m in enumerate(self) if isinstance(m, dict) and m.get("role") == "user"),
            None,
        )
        if first_user_idx is None:
            return 0
        prefix_len = first_user_idx + 1
        discarded = len(self) - prefix_len
        if discarded <= 0:
            return 0
        self._full.append(build_discard_all_marker(prefix_len, reason=reason))
        kept = [copy.deepcopy(self[j]) for j in range(prefix_len)]
        list.clear(self)          # bypass the blocked override; visible reset only
        list.extend(self, kept)   # (full already carries the marker)
        self._maybe_check()
        return discarded

    # -- untracked copy (intentional): throwaway scratch for summaries -------- #
    def copy(self) -> List[Dict[str, Any]]:  # type: ignore[override]
        return list(self)

    # -- verification -------------------------------------------------------- #
    def replay_full(self) -> List[Dict[str, Any]]:
        return replay(self._full)

    def assert_consistent(self) -> None:
        reconstructed = replay(self._full)
        if reconstructed != list(self):
            raise AssertionError(
                "ConversationHistory drift: replay(full) != visible "
                f"(replay has {len(reconstructed)} msgs, visible has {len(self)})."
            )

    def _maybe_check(self) -> None:
        if _RUNTIME_CHECK:
            self.assert_consistent()

    # -- blocked mutators (would silently desync the append-only invariant) --- #
    def _blocked(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "ConversationHistory forbids in-place bulk/positional mutation "
            "(insert/remove/clear/sort/reverse/setitem/delitem); use append/pop/"
            "discard_all_to_first_user so the append-only log stays consistent."
        )

    insert = _blocked
    remove = _blocked
    clear = _blocked
    sort = _blocked
    reverse = _blocked
    __setitem__ = _blocked
    __delitem__ = _blocked
