import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Callable

import flet as ft
from pynput.keyboard import Key as PynputKey
from pynput.keyboard import Listener as KeyboardListener

from core import MidiEngine, MidiRoleAnalysis, MidiRoleAnalyzer, TrackSplitPlan
from gui import HarmonyGui, StatusLevel


class AppController:
    """Coordinate UI interactions, playlist flow, and engine callbacks."""

    DEFAULT_TRANSITION_GAP_SECONDS = 1.5
    STATUS_AUTO_CLEAR_SECONDS = 3.0
    INTERRUPT_POLL_SECONDS = 0.05
    PROGRESS_UI_INTERVAL_SECONDS = 0.08
    MIDI_ALLOWED_EXTENSIONS = ["mid", "midi"]

    def __init__(self, page: ft.Page):
        """Initialize controller state and connect all event pipelines."""
        self.page = page
        self.gui = HarmonyGui(page)
        self.engine = MidiEngine()

        self.current_midi_path: str | None = None
        self.current_track_index: int = -1
        self.playlist_paths: list[str] = []
        self.playback_mode: str = "normal"
        self.smart_split_enabled: bool = False
        self.transition_gap_seconds = self.DEFAULT_TRANSITION_GAP_SECONDS

        self.play_thread: threading.Thread | None = None
        self.is_playing = False
        self.stop_requested = threading.Event()
        self.role_analysis_cache: dict[str, MidiRoleAnalysis] = {}
        self._status_token = 0
        self._pending_progress: tuple[float, float] | None = None
        self._progress_pump_running = False
        self._hotkey_listener: KeyboardListener | None = None

        self._bind_gui_hooks()
        self._bind_engine_callbacks()
        self._start_emergency_stop_listener()
        if self._hotkey_listener is None:
            self._show_message(self._tr("msg_hotkey_unavailable"), "warning")

    # ----- Wiring -----
    def _bind_gui_hooks(self) -> None:
        """Bind GUI hooks to controller handlers."""
        self.gui.on_play_click = self._handle_play_click
        self.gui.on_import_click = self._handle_import_click
        self.gui.on_jitter_change = self._handle_jitter_change
        self.gui.on_stagger_change = self._handle_stagger_change
        self.gui.on_library_track_select = self._handle_library_track_select
        self.gui.on_library_play_click = self._handle_library_play_click
        self.gui.on_status_close = self._handle_status_close
        self.gui.on_prev_click = self._handle_prev_click
        self.gui.on_next_click = self._handle_next_click
        self.gui.on_play_mode_change = self._handle_play_mode_change
        self.gui.on_split_toggle = self._handle_split_toggle
        self.gui.set_play_mode(self.playback_mode)
        self.gui.set_smart_split_enabled(self.smart_split_enabled)

    def _bind_engine_callbacks(self) -> None:
        """Bind engine callbacks to UI-safe handlers."""
        # Playback state is owned by the controller to avoid flicker between tracks.
        self.engine.on_progress = self._queue_progress_update
        self.engine.on_error = lambda message: self._run_on_ui(self._show_message, self._tr("msg_engine_error", message), "error")

    # ----- Utilities -----
    def _tr(self, key: str, *args: object) -> str:
        """Translate a locale key via GUI language context."""
        return self.gui.t(key, *args)

    def _run_on_ui(self, fn: Callable[..., None], *args: object) -> None:
        """Execute function on UI thread when API is available."""
        call_from_thread = getattr(self.page, "call_from_thread", None)
        if callable(call_from_thread):
            call_from_thread(fn, *args)
            return

        run_task = getattr(self.page, "run_task", None)
        if callable(run_task):
            async def invoke() -> None:
                fn(*args)

            try:
                run_task(invoke)
                return
            except Exception:
                pass

        # Last-resort fallback for runtimes without thread-dispatch helpers.
        fn(*args)

    def _start_emergency_stop_listener(self) -> None:
        """Start global ESC listener so playback can stop without UI focus."""
        if self._hotkey_listener is not None:
            return

        def on_press(key: Any) -> None:
            if key == PynputKey.esc:
                self._request_stop("msg_hotkey_stopped")

        try:
            self._hotkey_listener = KeyboardListener(on_press=on_press)
            self._hotkey_listener.daemon = True
            self._hotkey_listener.start()
        except Exception:
            # Listener may fail due to system permissions; user should grant accessibility permission.
            self._hotkey_listener = None

    def _request_stop(self, message_key: str) -> None:
        """Request a graceful stop once and surface a localized status message."""
        if not self.is_playing:
            return

        if self.stop_requested.is_set():
            return

        self.stop_requested.set()
        self.engine.stop()
        self._run_on_ui(self._show_message, self._tr(message_key), "warning")

    # ----- GUI Event Handlers -----
    def _handle_play_click(self, _: Any) -> None:
        """Start/stop playlist playback from Play button."""
        if self.is_playing:
            self._request_stop("msg_stopping_playlist")
            return

        if not self.playlist_paths:
            self._show_message(self._tr("msg_import_at_least_one"), "warning")
            return

        if self.play_thread and self.play_thread.is_alive():
            self._show_message(self._tr("msg_thread_still_running"), "warning")
            return

        self.stop_requested.clear()
        self.is_playing = True
        self.gui.set_playing_state(True)
        self._show_message(self._tr("msg_switch_to_game", self.engine.PRE_ROLL_SECONDS), "warning")

        self.play_thread = threading.Thread(
            target=self._play_playlist_worker,
            daemon=True,
        )
        self.play_thread.start()

    def _handle_import_click(self, _: Any) -> None:
        """Open file picker for MIDI playlist import."""
        self.page.run_task(self._pick_midi_files)

    def _handle_jitter_change(self, value: float) -> None:
        """Propagate jitter tuning from UI to engine."""
        self.engine.jitter_stdev = max(0.0, value)

    def _handle_stagger_change(self, value: float) -> None:
        """Propagate stagger tuning from UI to engine."""
        self.engine.chord_stagger = max(0.0, value)

    def _handle_status_close(self, _: Any) -> None:
        """Handle manual close for persistent error status."""
        self._status_token += 1
        self.gui.clear_status_message()

    def _handle_play_mode_change(self, mode: str) -> None:
        """Apply playback mode selected from UI."""
        if mode not in {"normal", "repeat_one", "repeat_all"}:
            return
        self.playback_mode = mode
        self._refresh_track_navigation_state()

    def _handle_split_toggle(self, enabled: bool) -> None:
        """Enable or disable smart split master switch."""
        self.smart_split_enabled = bool(enabled)
        self.gui.set_smart_split_enabled(self.smart_split_enabled)
        if self.smart_split_enabled:
            self._show_message(self._tr("msg_smart_split_enabled"), "info")
            self.page.run_task(self._analyze_current_track)
        else:
            self._show_message(self._tr("msg_smart_split_disabled"), "warning")

    def _handle_prev_click(self, _: Any) -> None:
        """Select previous track in normal (non-loop) mode."""
        self._navigate_track(-1)

    def _handle_next_click(self, _: Any) -> None:
        """Select next track in normal (non-loop) mode."""
        self._navigate_track(1)

    def _current_track_index(self) -> int:
        """Return current track index in playlist, defaulting to head."""
        if not self.playlist_paths:
            return 0
        if 0 <= self.current_track_index < len(self.playlist_paths):
            return self.current_track_index
        if self.current_midi_path in self.playlist_paths:
            return self.playlist_paths.index(self.current_midi_path)
        return 0

    def _refresh_track_navigation_state(self) -> None:
        """Update previous/next enabled state based on current index and total tracks."""
        total = len(self.playlist_paths)
        idx = self._current_track_index()
        if total <= 1:
            can_prev = False
            can_next = False
        elif self.playback_mode == "repeat_all":
            can_prev = True
            can_next = True
        else:
            can_prev = idx > 0
            can_next = idx < total - 1
        self.gui.set_track_navigation_state(can_prev, can_next)

    def _navigate_track(self, step: int) -> None:
        """Move current track by step in normal mode and optionally restart playback."""
        if not self.playlist_paths:
            self._refresh_track_navigation_state()
            return

        current_idx = self._current_track_index()
        total = len(self.playlist_paths)

        if self.playback_mode == "repeat_all" and total > 1:
            target_idx = (current_idx + step) % total
        else:
            target_idx = current_idx + step

        if target_idx < 0 or target_idx >= total:
            self._refresh_track_navigation_state()
            return

        target_path = self.playlist_paths[target_idx]
        self._handle_library_track_select(target_path)

        if self.is_playing:
            self._request_stop("msg_stopping_playlist")
            self.page.run_task(self._restart_after_track_change)

    async def _restart_after_track_change(self) -> None:
        """Wait for current playback worker to stop, then restart from selected track."""
        thread = self.play_thread
        if thread and thread.is_alive():
            await asyncio.to_thread(thread.join)

        if not self.is_playing:
            self._handle_play_click(None)

    def _handle_library_track_select(self, track_path: str) -> None:
        """Select a track by index without reordering playlist."""
        if not track_path:
            return

        if track_path not in self.playlist_paths:
            return

        self.current_track_index = self.playlist_paths.index(track_path)
        self.current_midi_path = track_path

        self.gui.set_library_tracks(self.playlist_paths, current_track_path=self.current_midi_path)
        self.gui.set_playback_snapshot(0.0, 0.0, 0.0)

        first_name = Path(track_path).name
        subtitle = self._tr("msg_playlist_count", len(self.playlist_paths))
        self.gui.set_track_info(f"[{self.current_track_index + 1}/{len(self.playlist_paths)}] {first_name}", subtitle)
        self._refresh_track_navigation_state()
        self.page.run_task(self._analyze_current_track)

    def _handle_library_play_click(self, track_path: str) -> None:
        """Select a track from library and jump to play view."""
        self._handle_library_track_select(track_path)
        self.gui.show_play_view()

    # ----- Playback Worker -----
    def _play_playlist_worker(self) -> None:
        """Play tracks according to current playback mode."""
        if not self.playlist_paths:
            return

        had_error = False
        completed = False
        current_idx = self._current_track_index()

        try:
            while not self.stop_requested.is_set():
                midi_path = self.playlist_paths[current_idx]

                if self.stop_requested.is_set():
                    break

                if not Path(midi_path).is_file():
                    self._run_on_ui(self._show_message, self._tr("msg_engine_error", f"File not found: {midi_path}"), "error")
                    had_error = True
                    break

                self.current_midi_path = midi_path
                self.current_track_index = current_idx
                track_name = Path(midi_path).name
                display_title = f"[{self.current_track_index + 1}/{len(self.playlist_paths)}] {track_name}"
                self._run_on_ui(self.gui.set_track_info, display_title, midi_path)
                self._run_on_ui(self.gui.set_library_tracks, self.playlist_paths, self.current_midi_path)
                self._run_on_ui(self._refresh_track_navigation_state)

                split_plan = self._build_split_plan_for_path(midi_path)

                try:
                    self.engine.play(midi_path, split_plan=split_plan)
                except Exception as exc:
                    self._run_on_ui(self._show_message, self._tr("msg_engine_error", str(exc)), "error")
                    had_error = True
                    break

                if self.stop_requested.is_set():
                    break

                next_idx: int | None
                if self.playback_mode == "repeat_one":
                    next_idx = current_idx
                elif self.playback_mode == "repeat_all":
                    next_idx = (current_idx + 1) % len(self.playlist_paths)
                else:
                    next_idx = current_idx + 1 if current_idx + 1 < len(self.playlist_paths) else None

                if next_idx is None:
                    completed = True
                    break

                if self.playback_mode != "repeat_one":
                    self._run_on_ui(
                        self._show_message,
                        self._tr("msg_track_finished_next", self.current_track_index + 1, self.transition_gap_seconds),
                    )
                    if not self._interruptible_sleep(self.transition_gap_seconds):
                        break

                current_idx = next_idx

            if self.stop_requested.is_set():
                self._run_on_ui(self._show_message, self._tr("msg_playlist_stopped"), "warning")
            elif not had_error and completed:
                self._run_on_ui(self._show_message, self._tr("msg_playlist_completed"))
        finally:
            self._run_on_ui(self._set_idle_state)

    def _interruptible_sleep(self, seconds: float) -> bool:
        """Sleep in short intervals so stop requests can interrupt quickly."""
        end_time = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end_time:
            if self.stop_requested.is_set():
                return False
            time.sleep(self.INTERRUPT_POLL_SECONDS)
        return True

    def _set_idle_state(self) -> None:
        """Return controller/UI to idle state after playlist loop exits."""
        self.is_playing = False
        self.play_thread = None
        self.gui.set_playing_state(False)
        self._refresh_track_navigation_state()

    # ----- Import Flow -----
    async def _pick_midi_files(self) -> None:
        """Pick one or more MIDI files and initialize playlist metadata."""
        selected_paths = await asyncio.to_thread(self._pick_with_native_dialog)

        if not selected_paths:
            self._show_message(self._tr("msg_read_path_failed"), "error")
            return

        existing = list(self.playlist_paths)
        normalized_existing = {self._normalize_path_key(path): path for path in existing}

        appended: list[str] = []
        for raw_path in selected_paths:
            normalized_path = str(Path(raw_path).expanduser().resolve(strict=False))
            key = self._normalize_path_key(normalized_path)
            if key in normalized_existing:
                continue
            normalized_existing[key] = normalized_path
            appended.append(normalized_path)

        self.playlist_paths = [*existing, *appended]

        if self.current_midi_path not in self.playlist_paths:
            self.current_track_index = 0 if self.playlist_paths else -1
            self.current_midi_path = self.playlist_paths[0] if self.playlist_paths else None
        elif self.current_midi_path is not None:
            self.current_track_index = self.playlist_paths.index(self.current_midi_path)

        self.gui.set_library_tracks(self.playlist_paths, current_track_path=self.current_midi_path)

        current_index = 1
        if self.current_midi_path and self.current_midi_path in self.playlist_paths:
            current_index = self.playlist_paths.index(self.current_midi_path) + 1

        first_name = Path(self.current_midi_path).name if self.current_midi_path else ""
        subtitle = self._tr("msg_playlist_count", len(self.playlist_paths))
        self.gui.set_track_info(f"[{current_index}/{len(self.playlist_paths)}] {first_name}", subtitle)
        self.gui.set_progress(0.0)
        self.gui.set_time_labels(0.0, 0.0)
        self._refresh_track_navigation_state()
        self._show_message(self._tr("msg_imported_count", len(appended)))
        self.page.run_task(self._analyze_current_track)

    async def _analyze_current_track(self) -> None:
        """Analyze selected MIDI structure and surface split-readiness status."""
        midi_path = self.current_midi_path
        if not midi_path:
            return

        try:
            normalized = str(Path(midi_path).expanduser().resolve(strict=False))
            analysis = self.role_analysis_cache.get(normalized)
            if analysis is None:
                analysis = await asyncio.to_thread(MidiRoleAnalyzer.analyze_file, normalized)
                self.role_analysis_cache[normalized] = analysis
        except Exception as exc:
            self._show_message(self._tr("msg_engine_error", str(exc)), "error")
            return

        if not self.smart_split_enabled:
            return

        if analysis.structured:
            self._show_message(self._tr("msg_track_split_ready", analysis.confidence), "info")
        else:
            reason_text = self._tr(f"msg_track_split_reason_{analysis.reason}")
            self._show_message(self._tr("msg_track_split_disabled", reason_text), "warning")

    def _build_split_plan_for_path(self, midi_path: str) -> TrackSplitPlan:
        """Build runtime split plan from cached analysis and master toggle."""
        if not self.smart_split_enabled:
            return TrackSplitPlan(enabled=False, structured=False, allowed_roles=("keyboard", "bass", "guitar", "drum"), channel_role_map=tuple())

        normalized = str(Path(midi_path).expanduser().resolve(strict=False))
        analysis = self.role_analysis_cache.get(normalized)
        if analysis is None:
            try:
                analysis = MidiRoleAnalyzer.analyze_file(normalized)
                self.role_analysis_cache[normalized] = analysis
            except Exception:
                return TrackSplitPlan(enabled=False, structured=False, allowed_roles=("keyboard", "bass", "guitar", "drum"), channel_role_map=tuple())

        if not analysis.structured:
            return TrackSplitPlan(enabled=False, structured=False, allowed_roles=("keyboard", "bass", "guitar", "drum"), channel_role_map=analysis.channel_role_map)

        return TrackSplitPlan(
            enabled=True,
            structured=True,
            allowed_roles=("keyboard", "bass", "guitar", "drum"),
            channel_role_map=analysis.channel_role_map,
        )

    @staticmethod
    def _normalize_path_key(path: str) -> str:
        """Normalize a path for robust deduplication across case and separators."""
        return str(Path(path).expanduser().resolve(strict=False)).casefold()

    def _pick_with_native_dialog(self) -> list[str]:
        """Pick MIDI files via native dialog to avoid runtime-specific Flet picker issues."""
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            filetypes = [("MIDI Files", "*.mid *.midi"), ("All Files", "*.*")]
            selected = filedialog.askopenfilenames(
                title=self._tr("msg_pick_midi_dialog"),
                filetypes=filetypes,
            )
            root.destroy()
            return [str(path) for path in selected if isinstance(path, str) and path]
        except Exception:
            return []

    # ----- Engine/UI Sync -----
    def _queue_progress_update(self, elapsed_seconds: float, total_seconds: float) -> None:
        """Store latest progress and start a throttled UI pump if needed."""
        self._pending_progress = (elapsed_seconds, total_seconds)
        if self._progress_pump_running:
            return

        self._progress_pump_running = True
        self.page.run_task(self._progress_ui_pump)

    async def _progress_ui_pump(self) -> None:
        """Flush latest progress to UI at a limited frame rate."""
        try:
            while self.is_playing or self._pending_progress is not None:
                pending = self._pending_progress
                self._pending_progress = None
                if pending is not None:
                    elapsed_seconds, total_seconds = pending
                    progress = (elapsed_seconds / total_seconds) if total_seconds > 0 else 0.0
                    self.gui.set_playback_snapshot(progress, elapsed_seconds, total_seconds)
                await asyncio.sleep(self.PROGRESS_UI_INTERVAL_SECONDS)
        finally:
            self._progress_pump_running = False
            if self._pending_progress is not None and self.is_playing:
                self._progress_pump_running = True
                self.page.run_task(self._progress_ui_pump)

    # ----- Status Messaging -----
    def _show_message(self, message: str, level: StatusLevel = "info") -> None:
        """Show leveled status text; auto-clear non-error messages."""
        self.gui.set_status_message(message, level=level)
        self._status_token += 1
        token = self._status_token
        if level != "error" and message:
            self.page.run_task(self._auto_clear_status, token, self.STATUS_AUTO_CLEAR_SECONDS)

    async def _auto_clear_status(self, token: int, delay_seconds: float) -> None:
        """Clear non-error status after delay if still latest message."""
        await asyncio.sleep(max(0.0, delay_seconds))
        if token != self._status_token:
            return
        self.gui.clear_status_message()


def main(page: ft.Page) -> None:
    """Flet app bootstrap entry."""
    AppController(page)


if __name__ == "__main__":
    ft.run(main)
