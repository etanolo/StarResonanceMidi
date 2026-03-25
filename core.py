"""MIDI playback engine and track-role analysis primitives.

This module contains low-level playback behavior, keyboard output mapping,
and optional track/channel role analysis structures used by the controller.
"""

import random
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

import mido
from pynput.keyboard import Controller, Key

# ----- Constants -----
# fmt: off
OCTAVE_KEYS = [
    'z', '1', 'x', '2', 'c', 'v', '3', 'b', '4', 'n', '5', 'm',
    'a', '6', 's', '7', 'd', 'f', '8', 'g', '9', 'h', '0', 'j',
    'q', 'i', 'w', 'o', 'e', 'r', 'p', 't', '[', 'y', ']', 'u',
]
# fmt: on

KEYBIND_MAP: dict[str, str | Key] = {
    "Ctrl": Key.ctrl_l,
    "Shift": Key.shift_l,
    "-": "-",
    "=": "=",
}

# Abstract transition graph: action-based, independent of actual keybinds.
_TRANSITION_GRAPH: dict[str, list[tuple[str, str]]] = {
    "LOW": [("up", "BASE"), ("shift", "LOW_SHIFT")],
    "LOW_SHIFT": [("up", "SHIFT"), ("shift", "LOW")],
    "CTRL": [("up", "HIGH_CTRL"), ("shift", "SHIFT"), ("ctrl", "BASE")],
    "BASE": [("up", "HIGH"), ("down", "LOW"), ("shift", "SHIFT"), ("ctrl", "CTRL")],
    "SHIFT": [("down", "LOW_SHIFT"), ("shift", "BASE"), ("ctrl", "CTRL")],
    "HIGH_CTRL": [("down", "CTRL"), ("ctrl", "HIGH")],
    "HIGH": [("down", "BASE"), ("ctrl", "HIGH_CTRL")],
}

# State ranges: (min_note, max_note, offset)
STATE_RANGES: dict[str, tuple[int, int, int]] = {
    "LOW": (21, 47, 12),  # A0 is lowest
    "LOW_SHIFT": (24, 59, 24),
    "CTRL": (36, 71, 36),
    "BASE": (48, 83, 48),
    "SHIFT": (60, 95, 60),
    "HIGH_CTRL": (72, 107, 72),
    "HIGH": (84, 108, 84),  # C8 is highest
}

PlayStateCallback = Callable[[bool], None]
ProgressCallback = Callable[[float, float], None]
TrackInfoCallback = Callable[[str, str], None]
ErrorCallback = Callable[[str], None]
FinishCallback = Callable[[], None]

TrackRole = str


@dataclass(frozen=True)
class TrackRoleDecision:
    """Single-track role decision with confidence diagnostics."""

    track_index: int
    track_name: str
    role: TrackRole
    confidence: float
    margin: float
    conflict_ratio: float
    dominant_channel: int | None
    dominant_channel_ratio: float


@dataclass(frozen=True)
class MidiRoleAnalysis:
    """Whole-file role analysis summary."""

    midi_path: str
    structured: bool
    confidence: float
    reason: Literal["ok", "no_tracks", "low_confidence", "ambiguous", "high_conflict"]
    decisions: tuple[TrackRoleDecision, ...]
    channel_role_map: tuple[tuple[int, TrackRole], ...]


@dataclass(frozen=True)
class TrackSplitPlan:
    """Runtime split plan passed from controller to engine."""

    enabled: bool
    structured: bool
    allowed_roles: tuple[str, ...]
    channel_role_map: tuple[tuple[int, TrackRole], ...]


@dataclass(frozen=True)
class PlannedBatch:
    """Group of notes to press simultaneously in one state."""

    abs_time: float
    state: str
    notes: tuple[int, ...]  # MIDI note numbers


class MidiRoleAnalyzer:
    """Rule-based MIDI track role analyzer with safety gating."""

    # Balanced defaults for real-world downloaded MIDIs.
    MIN_CONFIDENCE = 0.48
    MIN_MARGIN = 0.12
    MAX_CONFLICT_RATIO = 0.55
    MAX_AMBIGUOUS_TRACK_RATIO = 0.60
    MAX_HIGH_CONFLICT_TRACK_RATIO = 0.60

    DRUM_KEYWORDS = {
        "drum",
        "perc",
        "percussion",
        "kick",
        "snare",
        "hihat",
        "tom",
        "cymbal",
    }
    BASS_KEYWORDS = {"bass", "contra", "upright", "sub"}
    GUITAR_KEYWORDS = {"guitar", "gt", "lead", "rhythm", "acoustic", "electric"}
    KEYBOARD_KEYWORDS = {"piano", "keys", "keyboard", "organ", "synth", "ep", "clav"}

    @classmethod
    def analyze_file(cls, midi_path: str) -> MidiRoleAnalysis:
        """Analyze MIDI tracks and decide whether split is reliable enough."""
        midi = mido.MidiFile(midi_path, clip=True)
        decisions: list[TrackRoleDecision] = []

        for idx, track in enumerate(midi.tracks):
            decision = cls._analyze_track(idx, track)
            if decision is not None:
                decisions.append(decision)

        if not decisions:
            return MidiRoleAnalysis(
                midi_path=midi_path,
                structured=False,
                confidence=0.0,
                reason="no_tracks",
                decisions=tuple(),
                channel_role_map=tuple(),
            )

        channel_role_scores: dict[int, dict[TrackRole, float]] = {}
        for decision in decisions:
            if decision.dominant_channel is None:
                continue
            per_channel = channel_role_scores.setdefault(decision.dominant_channel, {})
            weight = decision.confidence * max(
                0.0, min(1.0, decision.dominant_channel_ratio)
            )
            per_channel[decision.role] = per_channel.get(decision.role, 0.0) + weight

        channel_role_map: list[tuple[int, TrackRole]] = []
        for channel, role_scores in channel_role_scores.items():
            best_role = max(role_scores.items(), key=lambda item: item[1])[0]
            channel_role_map.append((channel, best_role))

        overall_conf = sum(d.confidence for d in decisions) / len(decisions)
        ambiguous_count = sum(1 for d in decisions if d.margin < cls.MIN_MARGIN)
        high_conflict_count = sum(
            1 for d in decisions if d.conflict_ratio > cls.MAX_CONFLICT_RATIO
        )
        ambiguous_ratio = ambiguous_count / len(decisions)
        high_conflict_ratio = high_conflict_count / len(decisions)

        if overall_conf < cls.MIN_CONFIDENCE:
            reason: Literal[
                "ok", "no_tracks", "low_confidence", "ambiguous", "high_conflict"
            ] = "low_confidence"
            structured = False
        elif ambiguous_ratio > cls.MAX_AMBIGUOUS_TRACK_RATIO:
            reason = "ambiguous"
            structured = False
        elif high_conflict_ratio > cls.MAX_HIGH_CONFLICT_TRACK_RATIO:
            reason = "high_conflict"
            structured = False
        else:
            reason = "ok"
            structured = True

        return MidiRoleAnalysis(
            midi_path=midi_path,
            structured=structured,
            confidence=max(0.0, min(1.0, overall_conf)),
            reason=reason,
            decisions=tuple(decisions),
            channel_role_map=tuple(sorted(channel_role_map, key=lambda item: item[0])),
        )

    @classmethod
    def _analyze_track(
        cls, track_index: int, track: mido.MidiTrack
    ) -> TrackRoleDecision | None:
        """Score one MIDI track and produce a role decision."""
        scores: dict[TrackRole, float] = {
            "keyboard": 0.15,
            "bass": 0.0,
            "guitar": 0.0,
            "drum": 0.0,
        }

        track_name = ""
        note_events = 0
        channel_hist: dict[int, int] = {}
        program_hist: dict[int, int] = {}
        note_values: list[int] = []

        for msg in track:
            if (
                getattr(msg, "is_meta", False)
                and getattr(msg, "type", "") == "track_name"
            ):
                track_name = str(getattr(msg, "name", "") or "")
                continue

            msg_type = getattr(msg, "type", "")
            if msg_type == "program_change":
                raw_pg = getattr(msg, "program", None)
                program = int(raw_pg) if raw_pg is not None else 0
                raw_ch = getattr(msg, "channel", None)
                channel = int(raw_ch) if raw_ch is not None else 0
                program_hist[program] = program_hist.get(program, 0) + 1
                channel_hist[channel] = channel_hist.get(channel, 0) + 1
            elif msg_type == "note_on" and int(getattr(msg, "velocity", 0) or 0) > 0:
                note_events += 1
                raw_ch = getattr(msg, "channel", None)
                channel = int(raw_ch) if raw_ch is not None else 0
                channel_hist[channel] = channel_hist.get(channel, 0) + 1
                note = getattr(msg, "note", None)
                if isinstance(note, int):
                    note_values.append(note)

        if note_events == 0 and not program_hist and not track_name.strip():
            return None

        # GM drum hint: channel 10 (index 9).
        total_channel_events = sum(channel_hist.values())
        drum_ratio = (
            (channel_hist.get(9, 0) / total_channel_events)
            if total_channel_events > 0
            else 0.0
        )
        scores["drum"] += 0.85 * drum_ratio

        dominant_channel: int | None = None
        dominant_channel_ratio = 0.0
        if total_channel_events > 0:
            dominant_channel, dominant_events = max(
                channel_hist.items(), key=lambda item: item[1]
            )
            dominant_channel_ratio = dominant_events / total_channel_events

        # Program-range hints.
        total_program_events = sum(program_hist.values())
        if total_program_events > 0:
            bass_hits = sum(v for p, v in program_hist.items() if 32 <= p <= 39)
            guitar_hits = sum(v for p, v in program_hist.items() if 24 <= p <= 31)
            keyboard_hits = sum(v for p, v in program_hist.items() if 0 <= p <= 7)

            scores["bass"] += 0.80 * (bass_hits / total_program_events)
            scores["guitar"] += 0.80 * (guitar_hits / total_program_events)
            scores["keyboard"] += 0.65 * (keyboard_hits / total_program_events)

        # Note-range hints from played notes.
        if note_values:
            avg_note = sum(note_values) / len(note_values)
            low_ratio = sum(1 for n in note_values if n <= 52) / len(note_values)
            high_ratio = sum(1 for n in note_values if n >= 72) / len(note_values)

            if avg_note <= 50 or low_ratio >= 0.55:
                scores["bass"] += 0.28
            if 52 <= avg_note <= 82:
                scores["keyboard"] += 0.10
            if 50 <= avg_note <= 76 and high_ratio < 0.35:
                scores["guitar"] += 0.12

        # Track-name keyword hints.
        name_tokens = cls._tokenize(track_name)
        if name_tokens & cls.DRUM_KEYWORDS:
            scores["drum"] += 0.35
        if name_tokens & cls.BASS_KEYWORDS:
            scores["bass"] += 0.35
        if name_tokens & cls.GUITAR_KEYWORDS:
            scores["guitar"] += 0.35
        if name_tokens & cls.KEYBOARD_KEYWORDS:
            scores["keyboard"] += 0.30

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_role, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = max(0.0, best_score - second_score)

        # Conflict ratio from role-bucket dispersion.
        role_buckets = {
            "bass": sum(v for p, v in program_hist.items() if 32 <= p <= 39),
            "guitar": sum(v for p, v in program_hist.items() if 24 <= p <= 31),
            "keyboard": sum(v for p, v in program_hist.items() if 0 <= p <= 7),
            "drum": channel_hist.get(9, 0),
        }
        bucket_values = sorted(role_buckets.values(), reverse=True)
        bucket_total = sum(role_buckets.values())
        if bucket_total <= 0:
            conflict_ratio = 0.0
        else:
            dominant = bucket_values[0]
            conflict_ratio = 1.0 - (dominant / bucket_total)

        # Penalize likely multi-instrument tracks.
        if total_program_events > 0:
            dominant_program_ratio = max(program_hist.values()) / total_program_events
            if len(program_hist) >= 3 and dominant_program_ratio < 0.60:
                conflict_ratio += 0.15
        if total_channel_events > 0 and dominant_channel_ratio < 0.55:
            conflict_ratio += 0.10

        conflict_ratio = max(0.0, min(1.0, conflict_ratio))

        stability_factor = 1.0 - (0.25 * conflict_ratio)
        confidence = max(0.0, min(1.0, best_score * stability_factor))
        normalized_name = track_name.strip() or f"Track {track_index + 1}"

        return TrackRoleDecision(
            track_index=track_index,
            track_name=normalized_name,
            role=best_role,
            confidence=confidence,
            margin=margin,
            conflict_ratio=conflict_ratio,
            dominant_channel=dominant_channel,
            dominant_channel_ratio=dominant_channel_ratio,
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _tokenize(name: str) -> set[str]:
        """Tokenize track name for lightweight keyword matching."""
        normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in name)
        return {token for token in normalized.split() if token}


class MidiEngine:
    """Convert MIDI events into keyboard actions with humanization."""

    PRE_ROLL_SECONDS = 4.0
    ACTION_DP_WEIGHT = 2

    def __init__(self):
        """Initialize state machine, tuning values, and callbacks."""
        self.keyboard = Controller()

        # Runtime stop signal.
        self._stop_event = threading.Event()

        # Humanization tuning (can be updated by GUI).
        self.jitter_stdev = 0.012
        self.chord_stagger = 0.025

        # Toggle-style sustain pedal state tracked by app logic.
        self.sustain_is_on = False

        # Key state machine.
        self.current_state = "BASE"
        self.hesitation_min = 0.03
        self.hesitation_max = 0.05

        # seconds within which notes are considered a chord
        self.chord_threshold = 0.008

        # Configurable keybinds: abstract action -> actual key.
        self._action_keys: dict[str, str | Key] = {
            "ctrl": Key.ctrl_l,
            "shift": Key.shift_l,
            "down": ",",
            "up": ".",
        }

        # Controller-facing callbacks.
        self.on_play_state_change: PlayStateCallback | None = None
        self.on_progress: ProgressCallback | None = None
        self.on_track_info: TrackInfoCallback | None = None
        self.on_error: ErrorCallback | None = None
        self.on_finished: FinishCallback | None = None

    # ----- Callback helpers -----
    def _safe_emit(self, callback: Callable[..., None] | None, *args: object) -> None:
        """Emit callback safely without crashing playback thread."""
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            # Keep engine robust even if UI callback fails.
            return

    def _emit_progress(self, elapsed: float, total: float) -> None:
        """Emit clamped progress values."""
        clamped_elapsed = max(0.0, elapsed)
        safe_total = max(0.0, total)
        self._safe_emit(self.on_progress, clamped_elapsed, safe_total)

    # ----- Timing and key safety -----
    def precise_sleep(self, duration: float) -> None:
        """Sleep with short spin tail for tighter timing precision."""
        if duration <= 0:
            return

        target_time = time.perf_counter() + duration

        # Sleep most of the duration, then spin-wait the final milliseconds.
        sleep_time = duration - 0.015
        if sleep_time > 0:
            time.sleep(sleep_time)

        while time.perf_counter() < target_time:
            if self._stop_event.is_set():
                break
            pass

    def release_all_keys(self) -> None:
        """Release all potentially pressed keys and reset state."""
        self.keyboard.release(Key.space)
        for k in self._action_keys.values():
            self.keyboard.release(k)

        for k in OCTAVE_KEYS:
            self.keyboard.release(k)

    def prime_sustain_pedal(self) -> None:
        """Enable toggle-style sustain pedal only when currently off."""
        if self.sustain_is_on:
            return

        self.keyboard.release(Key.space)
        self.precise_sleep(0.02)
        self.keyboard.press(Key.space)
        self.precise_sleep(0.05)
        self.keyboard.release(Key.space)
        self.sustain_is_on = True

    # ----- State machine and note mapping -----
    def set_keybind(self, action: str, key: str | Key) -> None:
        """Update one keybind by action name."""
        if action in self._action_keys:
            self._action_keys[action] = key

    @classmethod
    @lru_cache(maxsize=None)
    def _bfs_actions(cls, source: str, target: str) -> tuple[str, ...]:
        """Return shortest abstract-action sequence from source to target state."""
        if source == target:
            return ()
        queue: list[tuple[str, list[str]]] = [(source, [])]
        visited = {source}
        while queue:
            state, actions = queue.pop(0)
            for action, nxt in _TRANSITION_GRAPH[state]:
                new_actions = actions + [action]
                if nxt == target:
                    return tuple(new_actions)
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, new_actions))
        return ()

    def switch_state(self, target: str) -> None:
        """Switch keyboard state using shortest action sequence, mapping to actual keys."""
        if self.current_state == target:
            return
        for action in self._bfs_actions(self.current_state, target):
            self.keyboard.tap(self._action_keys[action])
            self.precise_sleep(random.uniform(self.hesitation_min, self.hesitation_max))
        self.current_state = target

    def humanized_press(self, midi_note: int) -> None:
        """Press mapped key for a MIDI note, preferring current state or closest reachable."""
        # If current state can handle this note, stay in it.
        target_state: str | None = None
        offset = 0
        cur = STATE_RANGES.get(self.current_state)
        if cur is not None:
            lo, hi, off = cur
            if lo <= midi_note <= hi and 0 <= midi_note - off < len(OCTAVE_KEYS):
                target_state = self.current_state
                offset = off

        # Otherwise pick the closest reachable state by BFS distance.
        # Ties broken by preferring primary states over compound (LOW_SHIFT, HIGH_CTRL),
        # this prevents states only switching between LOW_SHIFT (C1-C3) and SHIFT (C4-C6),
        # because the bass and treble are typically distributed across these two non-overlapping regions, causing rapid switching.
        if target_state is None:
            best_dist = 999
            best_compound = True
            for state, (lo, hi, off) in STATE_RANGES.items():
                if lo <= midi_note <= hi and 0 <= midi_note - off < len(OCTAVE_KEYS):
                    is_compound = state in {"LOW_SHIFT", "HIGH_CTRL"}
                    dist = len(self._bfs_actions(self.current_state, state))
                    if dist < best_dist or (
                        dist == best_dist and best_compound and not is_compound
                    ):
                        best_dist = dist
                        best_compound = is_compound
                        target_state = state
                        offset = off

        if target_state is None:
            return

        # Dynamic hold time: shorter for high notes.
        # Not necessary when force prime sustain pedal
        # if midi_note > 72:
        #     p_min, p_max = 0.02, 0.05
        # else:
        #     p_min, p_max = 0.05, 0.10

        self.switch_state(target_state)

        key_idx = midi_note - offset
        if 0 <= key_idx < len(OCTAVE_KEYS):
            key = OCTAVE_KEYS[key_idx]
            self.keyboard.press(key)
            self.precise_sleep(random.uniform(self.hesitation_min, self.hesitation_max))
            self.keyboard.release(key)

    # ----- Whole-song pre-analysis -----
    def _precompute_note_plan(
        self,
        mid: mido.MidiFile,
        split_plan: TrackSplitPlan | None = None,
    ) -> list[PlannedBatch]:
        """Pre-analyze entire MIDI to find optimal state+batch sequence.

        Cost model:
        - Each batch (simultaneous key press group) = 1 unit cost.
        - Each BFS state-switch step = ACTION_DP_WEIGHT units cost.
        Same-state notes within a chord are batched together.
        State-level bitmask DP (7 bits) finds optimal intra-chord ordering.
        """
        # ── 1. Collect note_on events with absolute time in seconds ──
        raw: list[tuple[float, int]] = []  # (abs_time_seconds, midi_note)
        abs_seconds = 0.0
        current_tempo = 500_000  # default: 120 BPM
        ticks_per_beat = mid.ticks_per_beat or 480
        for msg in mido.merge_tracks(mid.tracks):
            delta_ticks = int(getattr(msg, "time", 0) or 0)
            abs_seconds += mido.tick2second(delta_ticks, ticks_per_beat, current_tempo)
            if getattr(msg, "type", "") == "set_tempo":
                current_tempo = int(getattr(msg, "tempo", 500_000) or 500_000)
                continue
            if split_plan and split_plan.enabled and split_plan.structured:
                if self._should_skip_message_by_role(msg, split_plan):
                    continue
            msg_type = getattr(msg, "type", None)
            msg_velocity = getattr(msg, "velocity", 0)
            msg_note = getattr(msg, "note", None)
            if msg_type == "note_on" and msg_velocity > 0 and isinstance(msg_note, int):
                raw.append((abs_seconds, msg_note))

        if not raw:
            return []

        # ── 2. State bookkeeping ──
        states = list(STATE_RANGES.keys())
        state_count = len(states)
        state_idx = {s: i for i, s in enumerate(states)}
        INF = float("inf")

        def _valid_si(note: int) -> int:
            """Return bitmask of valid state indices for *note*."""
            mask = 0
            for s, (lo, hi, off) in STATE_RANGES.items():
                if lo <= note <= hi and 0 <= note - off < len(OCTAVE_KEYS):
                    mask |= 1 << state_idx[s]
            return mask

        # BFS cost matrix (state_index * state_index)
        bfs = [[0] * state_count for _ in range(state_count)]
        for a in range(state_count):
            for b in range(state_count):
                bfs[a][b] = len(self._bfs_actions(states[a], states[b]))

        # ── 3. Build chord groups ──
        groups: list[list[int]] = [[0]]
        for i in range(1, len(raw)):
            if raw[i][0] - raw[groups[-1][0]][0] <= self.chord_threshold:
                groups[-1].append(i)
            else:
                groups.append([i])

        # ChordNote = (abs_time, midi_note, valid_state_bitmask)
        chords: list[list[tuple[float, int, int]]] = []
        for g in groups:
            ch: list[tuple[float, int, int]] = []
            for ri in g:
                vmask = _valid_si(raw[ri][1])
                if vmask:
                    ch.append((raw[ri][0], raw[ri][1], vmask))
            if ch:
                chords.append(ch)

        if not chords:
            return []

        n_chords = len(chords)
        state_mask = (1 << state_count) - 1  # 7-bit full mask

        # ── 4. Forward DP (state-level bitmask per chord) ──
        # dp[s] = min total cost to play everything so far, ending in state s
        dp = [INF] * state_count
        dp[state_idx["BASE"]] = 0

        # Per-chord reconstruction info
        par_tables: list[tuple] = [None] * n_chords  # type: ignore[list-item]

        for c, chord in enumerate(chords):
            # Compute per-note state masks and useful state set
            note_masks = [cn[2] for cn in chord]  # bitmask per note
            useful = 0
            for nm in note_masks:
                useful |= nm
            useful_list = [si for si in range(state_count) if useful & (1 << si)]

            if len(useful_list) == 1:
                # ── All notes fit a single state → one batch ──
                xs = useful_list[0]
                new_dp = [INF] * state_count
                par_note = [-1] * state_count
                for es in range(state_count):
                    if dp[es] >= INF:
                        continue
                    cost = dp[es] + bfs[es][xs] * self.ACTION_DP_WEIGHT + 1
                    if cost < new_dp[xs]:
                        new_dp[xs] = cost
                        par_note[xs] = es
                dp = new_dp
                par_tables[c] = ("one", par_note)

            else:
                # ── State-level bitmask DP ──
                cdp = [[INF] * state_count for _ in range(state_mask + 1)]
                par: list[list[tuple[int, int] | None]] = [
                    [None] * state_count for _ in range(state_mask + 1)
                ]

                for si in range(state_count):
                    cdp[0][si] = dp[si]

                for mask in range(state_mask + 1):
                    for si in range(state_count):
                        if cdp[mask][si] >= INF:
                            continue
                        for t in useful_list:
                            if mask & (1 << t):
                                continue
                            nm = mask | (1 << t)
                            cost = (
                                cdp[mask][si] + bfs[si][t] * self.ACTION_DP_WEIGHT + 1
                            )
                            if cost < cdp[nm][t]:
                                cdp[nm][t] = cost
                                par[nm][t] = (mask, si)

                # Collect best exits from any covering mask
                new_dp = [INF] * state_count
                best_mask = [0] * state_count
                for mask in range(state_mask + 1):
                    if all(mask & nm for nm in note_masks):
                        for si in range(state_count):
                            if cdp[mask][si] < new_dp[si]:
                                new_dp[si] = cdp[mask][si]
                                best_mask[si] = mask
                dp = new_dp
                par_tables[c] = ("batch", par, best_mask, note_masks)

        # ── 5. Find best ending state ──
        best_end = min(range(state_count), key=lambda i: dp[i])
        if dp[best_end] >= INF:
            return []

        # ── 6. Backward reconstruction ──
        result_rev: list[PlannedBatch] = []
        cur = best_end

        for c in range(n_chords - 1, -1, -1):
            chord = chords[c]
            tag = par_tables[c][0]

            if tag == "one":
                _, par_note = par_tables[c]
                result_rev.append(
                    PlannedBatch(
                        abs_time=chord[0][0],
                        state=states[cur],
                        notes=tuple(cn[1] for cn in chord),
                    )
                )
                cur = par_note[cur]

            elif tag == "batch":
                _, par, bm, note_masks_c = par_tables[c]
                mask = bm[cur]
                si = cur
                path: list[int] = []
                while mask != 0:
                    info = par[mask][si]
                    if info is None:
                        break
                    prev_mask, prev_si = info
                    path.append(si)
                    mask, si = prev_mask, prev_si
                path.reverse()
                entry_si = si

                # Assign notes to states in visit order
                assigned = set()
                batches_fwd: list[PlannedBatch] = []
                for s in path:
                    batch_notes: list[int] = []
                    for i, cn in enumerate(chord):
                        if i not in assigned and (note_masks_c[i] & (1 << s)):
                            batch_notes.append(cn[1])
                            assigned.add(i)
                    if batch_notes:
                        batches_fwd.append(
                            PlannedBatch(
                                abs_time=chord[0][0],
                                state=states[s],
                                notes=tuple(batch_notes),
                            )
                        )
                # Append reversed for result_rev (will be flipped later)
                result_rev.extend(reversed(batches_fwd))
                cur = entry_si

        result_rev.reverse()
        return result_rev

    def _play_planned_batch(self, batch: PlannedBatch) -> None:
        """Play a batch of notes after switching state.

        When chord_stagger == 0: press all keys simultaneously, wait once, release all.
        When chord_stagger != 0: press each key sequentially with stagger delay.
        """

        # already in the correct state because of pre-switching in main loop
        # keep this here just in case
        self.switch_state(batch.state)

        _, _, off = STATE_RANGES[batch.state]
        keys: list[str] = []
        for midi_note in batch.notes:
            key_idx = midi_note - off
            if 0 <= key_idx < len(OCTAVE_KEYS):
                keys.append(OCTAVE_KEYS[key_idx])

        if self.chord_stagger == 0:
            for key in keys:
                self.keyboard.press(key)
            self.precise_sleep(random.uniform(self.hesitation_min, self.hesitation_max))
            for key in keys:
                self.keyboard.release(key)
        else:
            for key in keys:
                self.keyboard.press(key)
                self.precise_sleep(
                    random.uniform(self.hesitation_min, self.hesitation_max)
                )
                self.keyboard.release(key)
                if key is not keys[-1]:
                    self.precise_sleep(random.uniform(0.0, self.chord_stagger))

    # ----- Playback lifecycle -----
    def play(self, midi_path: str, split_plan: TrackSplitPlan | None = None) -> None:
        """Blocking playback loop intended to run in a worker thread."""
        self._stop_event.clear()
        self._safe_emit(self.on_play_state_change, True)

        try:
            mid = mido.MidiFile(midi_path, clip=True)
        except Exception as e:
            err = str(e)
            self._safe_emit(self.on_error, err)
            self._safe_emit(self.on_play_state_change, False)
            self._safe_emit(self.on_finished)
            return

        track_title = Path(midi_path).name
        self._safe_emit(self.on_track_info, track_title, midi_path)

        total_duration = float(getattr(mid, "length", 0.0) or 0.0)
        self._emit_progress(0.0, total_duration)

        # Pre-compute optimal state plan for the entire song.
        note_plan = self._precompute_note_plan(mid, split_plan)

        # Brief pre-roll to allow user to focus game window.
        self.precise_sleep(self.PRE_ROLL_SECONDS)

        if self._stop_event.is_set():
            self._safe_emit(self.on_play_state_change, False)
            self._safe_emit(self.on_finished)
            return

        self.prime_sustain_pedal()

        try:
            start_time = time.perf_counter()
            for i, batch in enumerate(note_plan):
                if self._stop_event.is_set():
                    break

                # pre switch to correct state for the upcoming batch for more accurate timing
                self.switch_state(batch.state)

                # Add jitter / stagger offset.
                if i > 0 and batch.abs_time > note_plan[i - 1].abs_time + 0.001:
                    jitter = random.gauss(0, self.jitter_stdev)
                else:
                    jitter = random.uniform(0.0, self.chord_stagger)

                press_deadline = start_time + batch.abs_time + jitter
                wait = press_deadline - time.perf_counter()
                if wait > 0:
                    self.precise_sleep(wait)

                elapsed = time.perf_counter() - start_time
                self._emit_progress(elapsed, total_duration)

                self._play_planned_batch(batch)

        finally:
            self.release_all_keys()
            self.switch_state("BASE")
            self._emit_progress(total_duration, total_duration)
            self._safe_emit(self.on_play_state_change, False)
            self._safe_emit(self.on_finished)

    def stop(self) -> None:
        """Request stop for the running playback loop."""
        self._stop_event.set()

    @staticmethod
    def _resolve_message_role(msg: object, split_plan: TrackSplitPlan) -> str:
        """Resolve a message split key using channel mapping and fallback key."""
        raw_ch = getattr(msg, "channel", None)
        channel = int(raw_ch) if raw_ch is not None else -1
        channel_role_map = dict(split_plan.channel_role_map)
        mapped = channel_role_map.get(channel)
        if isinstance(mapped, str) and mapped:
            return mapped
        if channel >= 0:
            return f"ch:{channel + 1}"
        return "unknown"

    def _should_skip_message_by_role(
        self, msg: object, split_plan: TrackSplitPlan
    ) -> bool:
        """Return True when note events should be suppressed by active role filter."""
        msg_type = getattr(msg, "type", "")
        if msg_type != "note_on":
            return False

        velocity = int(getattr(msg, "velocity", 0) or 0)
        if velocity <= 0:
            return False

        role = self._resolve_message_role(msg, split_plan)
        return role not in set(split_plan.allowed_roles)
