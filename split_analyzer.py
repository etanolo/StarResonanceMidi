"""Channel split analysis for MIDI playback.

Implements a music21-first channel classifier with lightweight filtering and
target selection tuned for stable in-game playback control.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import mido

from instrument_classifier import InstrumentClassifier
from split_params import ParamsBundle, SplitClass, get_params


@dataclass(frozen=True)
class ChannelFeatureSummary:
    channel_1_based: int
    note_count: int
    min_pitch: int | None
    max_pitch: int | None
    avg_pitch: float | None
    low_pitch_ratio: float
    high_pitch_ratio: float
    drum_pitch_ratio: float
    program_hist: dict[int, int]
    name_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ChannelDecision:
    channel_1_based: int
    label: str
    split_class: SplitClass
    confidence: float
    margin: float
    note_count: int
    score_map: dict[SplitClass, float]


@dataclass(frozen=True)
class SplitAnalysisResult:
    midi_path: str
    decisions: tuple[ChannelDecision, ...]
    selected_targets: tuple[str, ...]


class SplitAnalyzer:
    """Analyze MIDI channels and produce stable split targets for playback."""

    def __init__(self, params: ParamsBundle | None = None) -> None:
        self.params = params or get_params("coherence_first")
        self.classifier = InstrumentClassifier()

    def analyze_file(self, midi_path: str, max_targets: int | None = None) -> SplitAnalysisResult:
        midi = mido.MidiFile(midi_path, clip=True)
        channel_features = self._collect_channel_features(midi)
        decisions = tuple(self._classify_channel(feature) for feature in channel_features)
        feature_map = {feature.channel_1_based: feature for feature in channel_features}

        limit = max_targets if max_targets is not None else self.params.limits.default_max_outputs
        selected = self._select_targets(decisions, feature_map, limit)
        return SplitAnalysisResult(
            midi_path=str(Path(midi_path).expanduser().resolve(strict=False)),
            decisions=decisions,
            selected_targets=tuple(selected),
        )

    def _collect_channel_features(self, midi: mido.MidiFile) -> list[ChannelFeatureSummary]:
        """Collect lightweight per-channel statistics for classification and ranking."""
        channel_notes: dict[int, list[int]] = {}
        channel_counts: dict[int, int] = {}
        channel_programs: dict[int, dict[int, int]] = {}
        channel_name_tokens: dict[int, set[str]] = {}

        for track in midi.tracks:
            track_name = ""
            track_channels: set[int] = set()
            for msg in track:
                if getattr(msg, "is_meta", False) and getattr(msg, "type", "") == "track_name":
                    track_name = str(getattr(msg, "name", "") or "")
                    continue
                if getattr(msg, "type", "") != "note_on":
                    if getattr(msg, "type", "") == "program_change":
                        raw_ch = getattr(msg, "channel", None)
                        channel = int(raw_ch) if raw_ch is not None else -1
                        raw_pg = getattr(msg, "program", None)
                        program = int(raw_pg) if raw_pg is not None else -1
                        if channel >= 0 and program >= 0:
                            track_channels.add(channel)
                            program_hist = channel_programs.setdefault(channel, {})
                            program_hist[program] = program_hist.get(program, 0) + 1
                    continue
                if int(getattr(msg, "velocity", 0) or 0) <= 0:
                    continue
                raw_ch = getattr(msg, "channel", None)
                channel = int(raw_ch) if raw_ch is not None else -1
                raw_note = getattr(msg, "note", None)
                note = int(raw_note) if raw_note is not None else -1
                if channel < 0 or note < 0:
                    continue
                track_channels.add(channel)
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
                channel_notes.setdefault(channel, []).append(note)

            if track_name.strip():
                tokens = self._tokenize(track_name)
                for channel in track_channels:
                    channel_name_tokens.setdefault(channel, set()).update(tokens)

        output: list[ChannelFeatureSummary] = []
        for channel, notes in channel_notes.items():
            if not notes:
                continue
            low_ratio = sum(1 for n in notes if n <= 52) / len(notes)
            high_ratio = sum(1 for n in notes if n >= 72) / len(notes)
            drum_ratio = sum(1 for n in notes if 35 <= n <= 81) / len(notes)
            output.append(
                ChannelFeatureSummary(
                    channel_1_based=channel + 1,
                    note_count=channel_counts.get(channel, 0),
                    min_pitch=min(notes),
                    max_pitch=max(notes),
                    avg_pitch=sum(notes) / len(notes),
                    low_pitch_ratio=low_ratio,
                    high_pitch_ratio=high_ratio,
                    drum_pitch_ratio=drum_ratio,
                    program_hist=dict(channel_programs.get(channel, {})),
                    name_tokens=tuple(sorted(channel_name_tokens.get(channel, set()))),
                )
            )

        output.sort(key=lambda item: item.note_count, reverse=True)
        return output

    def _classify_channel(self, feature: ChannelFeatureSummary) -> ChannelDecision:
        """Classify one channel with a simple music21-first strategy.

        Rules:
        1. Channel 10 is fixed to drum (handled in classifier).
        2. Other channels use music21 program mapping.
        3. If unresolved, mark as unknown and keep it visible.
        """
        program = 0
        if feature.program_hist:
            program = max(feature.program_hist.items(), key=lambda x: x[1])[0]

        best, confidence = self.classifier.classify_channel(feature.channel_1_based, program)

        score_map: dict[SplitClass, float] = {
            "drum": 0.0,
            "bass": 0.0,
            "guitar": 0.0,
            "keys": 0.0,
            "unknown": 0.0,
        }
        score_map[best] = max(0.0, min(1.0, confidence))

        # Keep sorting deterministic when confidence ties happen.
        margin = score_map[best]

        return ChannelDecision(
            channel_1_based=feature.channel_1_based,
            label=f"Channel {feature.channel_1_based}",
            split_class=best,
            confidence=confidence,
            margin=margin,
            note_count=feature.note_count,
            score_map=score_map,
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return {token for token in normalized.split() if token}

    def _select_targets(
        self,
        decisions: tuple[ChannelDecision, ...],
        feature_map: dict[int, ChannelFeatureSummary],
        limit: int,
    ) -> list[str]:
        """Select split targets with light filtering and optional intra-class seeding."""
        hard_limit = max(0, min(limit, self.params.limits.hard_max_outputs))
        if hard_limit <= 0:
            return []

        guarded = self._apply_continuity_guard(decisions)
        if not guarded:
            return []

        seed_channels = self._seed_channels_by_intra_class(guarded, feature_map)

        ordered = sorted(
            guarded,
            key=lambda d: (d.confidence, d.margin, d.note_count),
            reverse=True,
        )

        selected_channels: list[int] = []
        for channel in seed_channels:
            if channel not in selected_channels:
                selected_channels.append(channel)

        for decision in ordered:
            if decision.channel_1_based not in selected_channels:
                selected_channels.append(decision.channel_1_based)
            if len(selected_channels) >= hard_limit:
                break

        min_keep = min(self.params.limits.min_outputs, len(ordered))
        if len(selected_channels) < min_keep:
            for decision in ordered:
                if decision.channel_1_based not in selected_channels:
                    selected_channels.append(decision.channel_1_based)
                if len(selected_channels) >= min_keep:
                    break

        return [f"ch:{channel}" for channel in selected_channels[:hard_limit]]

    def _apply_continuity_guard(self, decisions: tuple[ChannelDecision, ...]) -> list[ChannelDecision]:
        """Remove only tiny-note-ratio branches, without class-based bias."""
        total_notes = sum(decision.note_count for decision in decisions)
        if total_notes <= 0:
            return []

        kept: list[ChannelDecision] = []
        # Keep this lenient to avoid dropping legitimate sparse channels.
        min_ratio = max(0.01, self.params.continuity.min_segment_note_ratio * 0.5)
        for decision in decisions:
            note_ratio = decision.note_count / total_notes
            if note_ratio < min_ratio:
                continue

            kept.append(decision)

        if kept:
            return kept

        # Guarantee at least one channel when all branches are weak.
        best = max(decisions, key=lambda d: (d.confidence, d.margin, d.note_count))
        return [best]

    def _seed_channels_by_intra_class(
        self,
        decisions: list[ChannelDecision],
        feature_map: dict[int, ChannelFeatureSummary],
    ) -> list[int]:
        """Pick representative channels from same-class sub-groups when separable."""
        if not self.params.intra_class.enabled:
            return []

        seeds: list[int] = []
        classes: tuple[SplitClass, ...] = ("bass", "guitar", "keys", "drum")
        for split_class in classes:
            group = [d for d in decisions if d.split_class == split_class]
            if len(group) < 2:
                continue

            total_notes = sum(d.note_count for d in group)
            if total_notes < self.params.intra_class.min_notes:
                continue

            with_pitch = [d for d in group if feature_map.get(d.channel_1_based) and feature_map[d.channel_1_based].avg_pitch is not None]
            if len(with_pitch) < 2:
                continue

            with_pitch.sort(key=lambda d: feature_map[d.channel_1_based].avg_pitch or 0.0)
            mid = len(with_pitch) // 2
            low_group = with_pitch[:mid]
            high_group = with_pitch[mid:]
            if not low_group or not high_group:
                continue

            low_avg = sum((feature_map[d.channel_1_based].avg_pitch or 0.0) for d in low_group) / len(low_group)
            high_avg = sum((feature_map[d.channel_1_based].avg_pitch or 0.0) for d in high_group) / len(high_group)
            separation = abs(high_avg - low_avg) / 127.0
            if separation < self.params.intra_class.min_separation:
                continue

            low_best = max(low_group, key=lambda d: (d.confidence, d.margin, d.note_count))
            high_best = max(high_group, key=lambda d: (d.confidence, d.margin, d.note_count))
            seeds.append(low_best.channel_1_based)
            seeds.append(high_best.channel_1_based)

        return seeds
