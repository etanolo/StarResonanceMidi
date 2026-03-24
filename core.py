"""MIDI playback engine and track-role analysis primitives.

This module contains low-level playback behavior, keyboard output mapping,
and optional track/channel role analysis structures used by the controller.
"""

import mido
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pynput.keyboard import Controller, Key

# ----- Constants -----
OCTAVE_KEYS = [
    'z', '1', 'x', '2', 'c', 'v', '3', 'b', '4', 'n', '5', 'm',
    'a', '6', 's', '7', 'd', 'f', '8', 'g', '9', 'h', '0', 'j',
    'q', 'i', 'w', 'o', 'e', 'r', 'p', 't', '[', 'y', ']', 'u'
]

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


class MidiRoleAnalyzer:
    """Rule-based MIDI track role analyzer with safety gating."""

    # Balanced defaults for real-world downloaded MIDIs.
    MIN_CONFIDENCE = 0.48
    MIN_MARGIN = 0.12
    MAX_CONFLICT_RATIO = 0.55
    MAX_AMBIGUOUS_TRACK_RATIO = 0.60
    MAX_HIGH_CONFLICT_TRACK_RATIO = 0.60

    DRUM_KEYWORDS = {"drum", "perc", "percussion", "kick", "snare", "hihat", "tom", "cymbal"}
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
            weight = decision.confidence * max(0.0, min(1.0, decision.dominant_channel_ratio))
            per_channel[decision.role] = per_channel.get(decision.role, 0.0) + weight

        channel_role_map: list[tuple[int, TrackRole]] = []
        for channel, role_scores in channel_role_scores.items():
            best_role = max(role_scores.items(), key=lambda item: item[1])[0]
            channel_role_map.append((channel, best_role))

        overall_conf = sum(d.confidence for d in decisions) / len(decisions)
        ambiguous_count = sum(1 for d in decisions if d.margin < cls.MIN_MARGIN)
        high_conflict_count = sum(1 for d in decisions if d.conflict_ratio > cls.MAX_CONFLICT_RATIO)
        ambiguous_ratio = ambiguous_count / len(decisions)
        high_conflict_ratio = high_conflict_count / len(decisions)

        if overall_conf < cls.MIN_CONFIDENCE:
            reason: Literal["ok", "no_tracks", "low_confidence", "ambiguous", "high_conflict"] = "low_confidence"
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
    def _analyze_track(cls, track_index: int, track: mido.MidiTrack) -> TrackRoleDecision | None:
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
            if getattr(msg, "is_meta", False) and getattr(msg, "type", "") == "track_name":
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
        drum_ratio = (channel_hist.get(9, 0) / total_channel_events) if total_channel_events > 0 else 0.0
        scores["drum"] += 0.85 * drum_ratio

        dominant_channel: int | None = None
        dominant_channel_ratio = 0.0
        if total_channel_events > 0:
            dominant_channel, dominant_events = max(channel_hist.items(), key=lambda item: item[1])
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
    def _tokenize(name: str) -> set[str]:
        """Tokenize track name for lightweight keyword matching."""
        normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in name)
        return {token for token in normalized.split() if token}


class MidiEngine:
    """Convert MIDI events into keyboard actions with humanization."""

    PRE_ROLL_SECONDS = 4.0

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
        self.hesitation_min = 0.08
        self.hesitation_max = 0.20

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
        self.keyboard.release(Key.shift_l)
        self.keyboard.release(Key.ctrl_l)
        self.keyboard.release(',')
        self.keyboard.release('.')

        for k in OCTAVE_KEYS:
            self.keyboard.release(k)

    def reset_state_to_base(self) -> None:
        """Force octave/state toggles back to BASE after playback."""
        if self.current_state == "SHIFT":
            self.keyboard.tap(Key.shift_l)
        elif self.current_state == "CTRL":
            self.keyboard.tap(Key.ctrl_l)
        elif self.current_state == "HIGH":
            self.keyboard.tap(',')
        elif self.current_state == "LOW":
            self.keyboard.tap('.')
        self.current_state = "BASE"

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
    def switch_state(self, target: str) -> None:
        """Switch keyboard modifier state with brief humanized delays."""
        if self.current_state == target:
            return

        self.precise_sleep(random.uniform(self.hesitation_min, self.hesitation_max))

        # 1) Release current state.
        if self.current_state == "SHIFT":
            self.keyboard.tap(Key.shift_l)
        elif self.current_state == "CTRL":
            self.keyboard.tap(Key.ctrl_l)
        elif self.current_state == "HIGH":
            self.keyboard.tap(',')
        elif self.current_state == "LOW":
            self.keyboard.tap('.')

        self.precise_sleep(random.uniform(0.01, 0.03))
        self.current_state = "BASE"

        if target == "BASE":
            return

        # 2) Enter target state.
        if target == "SHIFT":
            self.keyboard.tap(Key.shift_l)
        elif target == "CTRL":
            self.keyboard.tap(Key.ctrl_l)
        elif target == "HIGH":
            self.keyboard.tap('.')
        elif target == "LOW":
            self.keyboard.tap(',')

        self.current_state = target
        self.precise_sleep(random.uniform(0.02, 0.06))

    def humanized_press(self, midi_note: int) -> None:
        """Press mapped key for a MIDI note with variable hold time."""
        target_state = "BASE"
        offset = 48

        if 48 <= midi_note <= 83:
            target_state = "BASE"
            offset = 48
        elif 60 <= midi_note <= 95:
            target_state = "SHIFT"
            offset = 60
        elif 36 <= midi_note <= 71:
            target_state = "CTRL"
            offset = 36
        elif 84 <= midi_note <= 108:
            target_state = "HIGH"
            offset = 84
        elif 21 <= midi_note <= 47:
            target_state = "LOW"
            offset = 12
        else:
            return

        # Dynamic hold time: shorter for high notes.
        if midi_note > 72:
            p_min, p_max = 0.02, 0.05
        else:
            p_min, p_max = 0.05, 0.10

        self.switch_state(target_state)
        
        key_idx = midi_note - offset
        if 0 <= key_idx < len(OCTAVE_KEYS):
            key = OCTAVE_KEYS[key_idx]
            self.keyboard.press(key)
            self.precise_sleep(random.uniform(p_min, p_max))
            self.keyboard.release(key)

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
        elapsed = 0.0
        self._emit_progress(elapsed, total_duration)

        # Brief pre-roll to allow user to focus game window.
        self.precise_sleep(self.PRE_ROLL_SECONDS)

        if self._stop_event.is_set():
            self._safe_emit(self.on_play_state_change, False)
            self._safe_emit(self.on_finished)
            return

        self.prime_sustain_pedal()

        try:
            for msg in mid.play():
                if self._stop_event.is_set():
                    break

                if split_plan and split_plan.enabled and split_plan.structured:
                    if self._should_skip_message_by_role(msg, split_plan):
                        continue

                # mido.MidiFile.play() already handles base timing sleeps.
                msg_time = float(getattr(msg, "time", 0.0) or 0.0)
                extra_delay = 0.0
                if msg_time > 0:
                    extra_delay = max(0.0, random.gauss(0, self.jitter_stdev))
                else:
                    extra_delay = random.uniform(0.002, self.chord_stagger)

                if extra_delay > 0:
                    self.precise_sleep(extra_delay)

                elapsed += msg_time + extra_delay
                self._emit_progress(elapsed, total_duration)

                msg_type = getattr(msg, "type", None)
                msg_velocity = getattr(msg, "velocity", 0)
                msg_note = getattr(msg, "note", None)
                if msg_type == "note_on" and msg_velocity > 0 and isinstance(msg_note, int):
                    self.humanized_press(msg_note)

        finally:
            self.release_all_keys()
            self.reset_state_to_base()
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

    def _should_skip_message_by_role(self, msg: object, split_plan: TrackSplitPlan) -> bool:
        """Return True when note events should be suppressed by active role filter."""
        msg_type = getattr(msg, "type", "")
        if msg_type != "note_on":
            return False

        velocity = int(getattr(msg, "velocity", 0) or 0)
        if velocity <= 0:
            return False

        role = self._resolve_message_role(msg, split_plan)
        return role not in set(split_plan.allowed_roles)