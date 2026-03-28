"""Flet UI layer for StarResonanceMidi.

This module builds interactive views and exposes typed hooks/update APIs
consumed by the application controller.
"""

import json
import locale as py_locale
from pathlib import Path
from typing import Any, Callable, Literal, cast

import flet as ft

from app_info import APP_NAME, APP_VERSION
from core import KEYBIND_MAP

FALLBACK_LOCALES: dict[str, dict[str, str]] = {"en": {}, "ja": {}, "zh": {}}


def load_locales(filepath: str = "locales.json") -> dict[str, dict[str, str]]:
    """Load locale mapping from JSON with safe fallback."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return FALLBACK_LOCALES
    except FileNotFoundError:
        print(f"Warning: File not found {filepath}, using fallback empty dictionary.")
        return FALLBACK_LOCALES
    except json.JSONDecodeError:
        print(f"Warning: {filepath} has invalid JSON format, please check the syntax.")
        return FALLBACK_LOCALES

LOCALES = load_locales()

# ----- Type aliases -----
UiEventHandler = Callable[[Any], None]
ValueChangeHandler = Callable[[float], None]
TrackSelectHandler = Callable[[str], None]
ModeChangeHandler = Callable[[str], None]
BoolChangeHandler = Callable[[bool], None]
RoleToggleHandler = Callable[[str, bool], None]
TracksRemoveHandler = Callable[[list[str]], None]
KeybindChangeHandler = Callable[[str, str], None]  # (action, display_label)
StatusLevel = Literal["info", "warning", "error"]
STATUS_COLOR_MAP: dict[StatusLevel, str] = {
    "info": ft.Colors.GREY_700,
    "warning": ft.Colors.AMBER_700,
    "error": ft.Colors.RED_700,
}


class StarResonanceMidiGui:
    """Build and update application UI components."""

    def __init__(self, page: ft.Page):
        """Initialize page setup, hooks, and view tree."""
        self.page = page
        self.page.title = APP_NAME
        self.page.theme_mode = ft.ThemeMode.LIGHT
        
        self.page.window.width = 1000
        self.page.window.height = 800
        self._apply_window_icon()
        self.page.theme = ft.Theme(color_scheme_seed="#375ca8")

        self.current_lang = self._detect_initial_language()
        
        # Hooks exposed for controller binding.
        self.on_play_click: UiEventHandler | None = None
        self.on_import_click: UiEventHandler | None = None
        self.on_jitter_change: ValueChangeHandler | None = None
        self.on_stagger_change: ValueChangeHandler | None = None
        self.on_library_track_select: TrackSelectHandler | None = None
        self.on_library_play_click: TrackSelectHandler | None = None
        self.on_library_remove_click: TracksRemoveHandler | None = None
        self.on_status_close: UiEventHandler | None = None
        self.on_prev_click: UiEventHandler | None = None
        self.on_next_click: UiEventHandler | None = None
        self.on_play_mode_change: ModeChangeHandler | None = None
        self.on_split_toggle: BoolChangeHandler | None = None
        self.on_split_role_toggle: RoleToggleHandler | None = None
        self.on_hesitation_min_change: ValueChangeHandler | None = None
        self.on_hesitation_max_change: ValueChangeHandler | None = None
        self.on_keybind_change: KeybindChangeHandler | None = None

        # Internal control refs used by update APIs.
        self.btn_play: ft.FloatingActionButton | None = None
        self.btn_prev: ft.IconButton | None = None
        self.btn_next: ft.IconButton | None = None
        self.btn_mode_normal: ft.Button | None = None
        self.btn_mode_repeat_one: ft.Button | None = None
        self.btn_mode_repeat_all: ft.Button | None = None
        self.switch_split_enabled: ft.Switch | None = None
        self.split_role_row: ft.Row | None = None
        self.current_play_mode: str = "normal"
        self.split_enabled: bool = True
        self.split_target_buttons: dict[str, ft.Button] = {}
        self.split_target_labels: dict[str, str] = {}
        self.enabled_split_roles: set[str] = set()
        self.split_targets_locked: bool = False
        self.progress_bar: ft.ProgressBar | None = None
        self.lbl_time_current: ft.Text | None = None
        self.lbl_time_total: ft.Text | None = None
        self.lbl_track_title: ft.Text | None = None
        self.lbl_track_sub: ft.Text | None = None
        self.lbl_status: ft.Text | None = None
        self.btn_status_close: ft.IconButton | None = None
        self.library_tracks: list[str] = []
        self.library_selected_paths: set[str] = set()
        self.current_track_path: str | None = None
        self.library_list_view: ft.ListView | None = None
        self.library_search_field: ft.TextField | None = None
        self.btn_library_remove: ft.Button | None = None
        self.btn_library_select_all: ft.Button | None = None
        self.btn_library_invert: ft.Button | None = None

        self.init_ui()
        self._apply_font_for_language()

    def _apply_window_icon(self) -> None:
        """Set desktop window icon from local assets when supported by runtime."""
        window_obj = getattr(self.page, "window", None)
        if window_obj is None or not hasattr(window_obj, "icon"):
            return

        base_dir = Path(__file__).resolve().parent
        icon_path = base_dir / "assets" / "icon.ico"
        if icon_path is None:
            return
        if not icon_path.exists():
            return

        try:
            window_obj.icon = str(icon_path)
        except Exception:
            # Ignore icon errors on unsupported runtimes.
            pass

    # ----- Public update APIs -----
    def set_playing_state(self, is_playing: bool) -> None:
        """Toggle play button icon between play/stop."""
        if self.btn_play:
            self.btn_play.icon = ft.Icons.STOP if is_playing else ft.Icons.PLAY_ARROW
        if self.switch_split_enabled is not None:
            self.switch_split_enabled.disabled = bool(is_playing)
        # Lock split-role toggles during playback.
        self.split_targets_locked = bool(is_playing)
        self._refresh_split_role_buttons()

    def set_track_info(self, title: str, subtitle: str = "") -> None:
        """Update current track title/subtitle."""
        if self.lbl_track_title and self.lbl_track_sub:
            self.lbl_track_title.value = title
            self.lbl_track_sub.value = subtitle
            self.page.update()

    def set_track_navigation_state(self, can_prev: bool, can_next: bool) -> None:
        """Enable or disable previous/next controls by playlist boundaries."""
        if self.btn_prev is not None:
            self.btn_prev.disabled = not can_prev
            self.btn_prev.tooltip = self.t("play_prev") if can_prev else self.t("play_prev_unavailable")
        if self.btn_next is not None:
            self.btn_next.disabled = not can_next
            self.btn_next.tooltip = self.t("play_next") if can_next else self.t("play_next_unavailable")
        self.page.update()

    def set_play_mode(self, mode: str) -> None:
        """Set current play mode and refresh mode button visual state."""
        allowed = {"normal", "repeat_one", "repeat_all"}
        self.current_play_mode = mode if mode in allowed else "normal"
        self._refresh_play_mode_buttons()

    def set_split_roles(self, target_labels: dict[str, str], enabled_roles: set[str]) -> None:
        """Sync split targets and enabled state from controller."""
        self.split_target_labels = dict(target_labels)
        self.enabled_split_roles = set(enabled_roles)
        self._refresh_split_role_buttons()

    def set_split_enabled(self, enabled: bool) -> None:
        """Sync split master toggle state from controller."""
        self.split_enabled = bool(enabled)
        if self.switch_split_enabled is not None:
            self.switch_split_enabled.value = self.split_enabled
        self._refresh_split_role_buttons()

    def _refresh_play_mode_buttons(self) -> None:
        """Highlight selected mode button and reset others."""
        button_map: dict[str, ft.Button | None] = {
            "normal": self.btn_mode_normal,
            "repeat_one": self.btn_mode_repeat_one,
            "repeat_all": self.btn_mode_repeat_all,
        }

        for mode_key, button in button_map.items():
            if button is None:
                continue
            is_selected = mode_key == self.current_play_mode
            button.style = ft.ButtonStyle(
                bgcolor=ft.Colors.PRIMARY if is_selected else ft.Colors.SURFACE,
                color=ft.Colors.WHITE if is_selected else ft.Colors.ON_SURFACE,
                side=ft.BorderSide(1, ft.Colors.PRIMARY),
            )

        self.page.update()

    def _refresh_split_role_buttons(self) -> None:
        """Refresh split-target button list and selected state."""
        if self.split_role_row is None:
            return

        controls: list[ft.Control] = [ft.Text(self.t("play_split_label"), size=12, color=ft.Colors.GREY_600)]
        self.split_target_buttons = {}

        def split_sort_key(target_key: str) -> tuple[int, str]:
            if target_key.startswith("ch:"):
                raw = target_key.split(":", 1)[1]
                if raw.isdigit():
                    return (int(raw), "")
            return (10_000, target_key)

        for target_key in sorted(self.split_target_labels.keys(), key=split_sort_key):
            label = self.split_target_labels[target_key]
            is_enabled = target_key in self.enabled_split_roles
            button = ft.Button(
                label,
                on_click=lambda e, k=target_key: self._handle_split_role_button_click(k),
                disabled=self.split_targets_locked or not self.split_enabled,
                style=ft.ButtonStyle(
                    bgcolor=ft.Colors.PRIMARY if is_enabled else ft.Colors.SURFACE,
                    color=ft.Colors.WHITE if is_enabled else ft.Colors.ON_SURFACE,
                    side=ft.BorderSide(1, ft.Colors.PRIMARY),
                ),
            )
            self.split_target_buttons[target_key] = button
            controls.append(button)

        self.split_role_row.controls = controls
        self.split_role_row.visible = self.split_enabled and bool(self.split_target_labels)
        self.page.update()

    def _handle_play_mode_button_click(self, mode: str) -> None:
        """Update UI mode selection and notify controller."""
        self.set_play_mode(mode)
        if self.on_play_mode_change:
            self.on_play_mode_change(mode)

    def _handle_split_role_button_click(self, role: str) -> None:
        """Toggle one role selection and notify controller."""
        currently_enabled = role in self.enabled_split_roles
        next_enabled = not currently_enabled
        if self.on_split_role_toggle:
            self.on_split_role_toggle(role, next_enabled)

    def _handle_split_toggle_change(self, enabled: bool) -> None:
        """Toggle split master switch and notify controller."""
        self.set_split_enabled(enabled)
        if self.on_split_toggle:
            self.on_split_toggle(enabled)

    def set_progress(self, progress: float) -> None:
        """Update progress bar value in range [0, 1]."""
        if self.progress_bar:
            clamped = max(0.0, min(1.0, progress))
            self.progress_bar.value = clamped
            self.page.update()

    def set_time_labels(self, current_seconds: float, total_seconds: float) -> None:
        """Update elapsed and total time labels."""
        if self.lbl_time_current and self.lbl_time_total:
            self.lbl_time_current.value = self.format_seconds(current_seconds)
            self.lbl_time_total.value = self.format_seconds(total_seconds)
            self.page.update()

    def set_playback_snapshot(self, progress: float, current_seconds: float, total_seconds: float) -> None:
        """Update progress and time labels in a single UI refresh."""
        has_progress = self.progress_bar is not None
        has_labels = self.lbl_time_current is not None and self.lbl_time_total is not None
        if not has_progress and not has_labels:
            return

        if self.progress_bar is not None:
            clamped = max(0.0, min(1.0, progress))
            self.progress_bar.value = clamped

        if self.lbl_time_current and self.lbl_time_total:
            self.lbl_time_current.value = self.format_seconds(current_seconds)
            self.lbl_time_total.value = self.format_seconds(total_seconds)

        self.page.update()

    def set_status_message(self, message: str, level: StatusLevel = "info") -> None:
        """Render leveled status text inside player card."""
        if self.lbl_status:
            self.lbl_status.value = message
            self.lbl_status.color = STATUS_COLOR_MAP.get(level, ft.Colors.GREY_700)
            if self.btn_status_close:
                self.btn_status_close.visible = level == "error" and bool(message)
            self.page.update()

    def clear_status_message(self) -> None:
        """Clear status row and hide close button."""
        if self.lbl_status:
            self.lbl_status.value = ""
            if self.btn_status_close:
                self.btn_status_close.visible = False
            self.page.update()

    def set_library_tracks(self, paths: list[str], current_track_path: str | None = None) -> None:
        """Update library list from imported MIDI paths."""
        self.library_tracks = list(paths)
        self.library_selected_paths &= set(self.library_tracks)
        if current_track_path is not None:
            self.current_track_path = current_track_path
        self._refresh_library_list()

    def _handle_library_select_toggle(self, track_path: str, selected: bool) -> None:
        """Toggle one track selection used by batch remove."""
        if selected:
            self.library_selected_paths.add(track_path)
        else:
            self.library_selected_paths.discard(track_path)
        self._refresh_library_action_state()

    def _get_filtered_library_tracks(self) -> list[str]:
        """Return tracks currently visible under search filter."""
        query = ""
        if self.library_search_field and isinstance(self.library_search_field.value, str):
            query = self.library_search_field.value.strip().lower()
        if not query:
            return list(self.library_tracks)
        return [path for path in self.library_tracks if query in Path(path).name.lower()]

    def _handle_library_select_all_click(self) -> None:
        """Select all currently visible tracks."""
        self.library_selected_paths.update(self._get_filtered_library_tracks())
        self._refresh_library_list()

    def _handle_library_invert_selection_click(self) -> None:
        """Invert selection for currently visible tracks only."""
        visible_tracks = self._get_filtered_library_tracks()
        visible_set = set(visible_tracks)
        selected_visible = self.library_selected_paths & visible_set
        self.library_selected_paths = (self.library_selected_paths - visible_set) | (visible_set - selected_visible)
        self._refresh_library_list()

    def _handle_library_remove_selected_click(self) -> None:
        """Request controller to remove selected tracks from playlist."""
        if self.on_library_remove_click:
            self.on_library_remove_click(list(self.library_selected_paths))

    def _refresh_library_action_state(self) -> None:
        """Enable or disable batch actions according to current selection."""
        if self.btn_library_remove is not None:
            self.btn_library_remove.disabled = len(self.library_selected_paths) == 0
        visible_tracks = self._get_filtered_library_tracks()
        has_visible = len(visible_tracks) > 0
        if self.btn_library_select_all is not None:
            self.btn_library_select_all.disabled = not has_visible
        if self.btn_library_invert is not None:
            self.btn_library_invert.disabled = not has_visible
        self.page.update()

    @staticmethod
    def format_seconds(seconds: float) -> str:
        """Format duration seconds into MM:SS."""
        safe = max(0.0, seconds)
        total = int(safe)
        minute, sec = divmod(total, 60)
        return f"{minute:02d}:{sec:02d}"

    # ----- Localization and theming -----
    def t(self, key: str, *args: object) -> str:
        """Translate key using current language dictionary."""
        current_dict = LOCALES.get(self.current_lang, LOCALES.get("en", {}))
        text = current_dict.get(key, key)
        if not isinstance(text, str):
            text = str(text)
        if args:
            return text.format(*args)
        return text

    def _detect_initial_language(self) -> str:
        """Detect initial UI language: zh/ja/en only, fallback to en."""
        candidates: list[str] = []

        locale_config = getattr(self.page, "locale_configuration", None)
        if locale_config is not None:
            current_locale = getattr(locale_config, "current_locale", None)
            if isinstance(current_locale, str):
                candidates.append(current_locale)
            elif current_locale is not None:
                lang_code = getattr(current_locale, "language_code", None)
                country_code = getattr(current_locale, "country_code", None)
                if isinstance(lang_code, str) and isinstance(country_code, str):
                    candidates.append(f"{lang_code}_{country_code}")
                if isinstance(lang_code, str):
                    candidates.append(lang_code)

        try:
            default_locale = py_locale.getdefaultlocale()[0]
        except Exception:
            default_locale = None
        if isinstance(default_locale, str):
            candidates.append(default_locale)

        try:
            current_locale = py_locale.getlocale()[0]
        except Exception:
            current_locale = None
        if isinstance(current_locale, str):
            candidates.append(current_locale)

        for raw in candidates:
            normalized = raw.replace("-", "_").lower()
            if normalized.startswith("zh"):
                return "zh"
            if normalized.startswith("ja"):
                return "ja"
            if normalized.startswith("en"):
                return "en"

        return "en"

    def toggle_theme(self, e: Any) -> None:
        """Switch light/dark theme mode."""
        self.page.theme_mode = ft.ThemeMode.DARK if e.control.value else ft.ThemeMode.LIGHT
        self.page.update()

    def change_language(self, e: Any) -> None:
        """Rebuild localized views after language selection changes."""
        lang_map = {"English": "en", "日本語": "ja", "简体中文": "zh"}
        new_lang = lang_map.get(e.control.value, "en")
        
        if self.current_lang != new_lang:
            self.current_lang = new_lang
            self._apply_font_for_language()
            self.play_view = self.build_play_view()
            self.library_view = self.build_library_view()
            self.settings_view = self.build_settings_view()
            
            current_index = self.nav_rail.selected_index
            if current_index == 0:
                self.content_area.content = self.play_view
            elif current_index == 1:
                self.content_area.content = self.library_view
            elif current_index == 2:
                self.content_area.content = self.settings_view
                
            self.nav_rail.destinations[0].label = self.t("nav_play")
            self.nav_rail.destinations[1].label = self.t("nav_library")
            self.nav_rail.destinations[2].label = self.t("nav_settings")
            
            self.page.update()

    def _apply_font_for_language(self) -> None:
        """Use system default fonts only for offline-friendly and lightweight builds."""
        self.page.theme = ft.Theme(color_scheme_seed="#375ca8")

    def show_play_view(self) -> None:
        """Navigate to play page programmatically."""
        self.nav_rail.selected_index = 0
        self.content_area.content = self.play_view
        self.page.update()

    # ----- Layout skeleton -----
    def init_ui(self) -> None:
        """Create root layout with rail navigation and content area."""
        self.play_view = self.build_play_view()
        self.library_view = self.build_library_view()
        self.settings_view = self.build_settings_view()

        self.content_area = ft.Container(content=self.play_view, expand=True, padding=20)

        self.nav_rail = ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=100,
            destinations=[
                ft.NavigationRailDestination(icon=ft.Icons.PLAY_CIRCLE_OUTLINE, selected_icon=ft.Icons.PLAY_CIRCLE_FILLED, label=self.t("nav_play")),
                ft.NavigationRailDestination(icon=ft.Icons.LIBRARY_MUSIC, selected_icon=ft.Icons.LIBRARY_MUSIC, label=self.t("nav_library")),
                ft.NavigationRailDestination(icon=ft.Icons.SETTINGS_OUTLINED, selected_icon=ft.Icons.SETTINGS, label=self.t("nav_settings")),
            ],
            on_change=self.handle_nav_change
        )

        self.page.add(ft.Row(controls=[self.nav_rail, ft.VerticalDivider(width=1), self.content_area], expand=True))

    def handle_nav_change(self, e: Any) -> None:
        """Swap content view when rail index changes."""
        index = e.control.selected_index
        if index == 0:
            self.content_area.content = self.play_view
        elif index == 1:
            self.content_area.content = self.library_view
        elif index == 2:
            self.content_area.content = self.settings_view
        self.page.update()

    # ----- View helpers -----
    def create_slider_row(self, title: str, description: str, slider_control: ft.Slider) -> ft.Column:
        """Build a titled slider row with live value text."""
        value_text = ft.Text(f"{slider_control.value:.3f}s", weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY)
        
        # Wrap original callback to keep UI value text and hook behavior in sync.
        original_on_change = slider_control.on_change

        def wrapped_on_change(e: Any) -> None:
            value_text.value = f"{e.control.value:.3f}s"
            self.page.update()
            if original_on_change:
                cast(Callable[[Any], None], original_on_change)(e)

        slider_control.on_change = wrapped_on_change

        return ft.Column(
            controls=[
                ft.Row(controls=[ft.Text(title, weight=ft.FontWeight.W_500), value_text], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Text(description, size=12, color=ft.Colors.GREY_600),
                slider_control
            ],
            spacing=5
        )

    def _refresh_library_list(self) -> None:
        """Render library list with optional search filter."""
        if self.library_list_view is None:
            return

        query = ""
        if self.library_search_field and isinstance(self.library_search_field.value, str):
            query = self.library_search_field.value.strip().lower()

        controls: list[ft.Control] = []
        for track_path in self.library_tracks:
            filename = Path(track_path).name
            if query and query not in filename.lower():
                continue

            is_current_track = track_path == self.current_track_path
            is_selected = track_path in self.library_selected_paths
            trailing_control: ft.Control
            if is_current_track:
                trailing_control = ft.IconButton(
                    icon=ft.Icons.PLAY_CIRCLE_FILLED,
                    icon_color=ft.Colors.PRIMARY,
                    tooltip=self.t("nav_play"),
                    on_click=lambda e, p=track_path: self.on_library_play_click(p) if self.on_library_play_click else None,
                )
            else:
                trailing_control = ft.Icon(ft.Icons.AUDIOTRACK, color=ft.Colors.PRIMARY)

            controls.append(
                ft.ListTile(
                    leading=ft.Checkbox(
                        value=is_selected,
                        on_change=lambda e, p=track_path: self._handle_library_select_toggle(p, bool(e.control.value)),
                    ),
                    title=ft.Text(f"▶ {filename}" if is_current_track else filename),
                    subtitle=ft.Text(track_path),
                    trailing=trailing_control,
                    on_click=lambda e, p=track_path: self.on_library_track_select(p) if self.on_library_track_select else None,
                )
            )

        self.library_list_view.controls = controls
        self._refresh_library_action_state()

    # ----- View builders -----
    def build_play_view(self) -> ft.Column:
        """Build player view with track card, controls, and tuning section."""
        self.lbl_track_title = ft.Text(self.t("play_empty_title"), size=20, weight=ft.FontWeight.BOLD)
        self.lbl_track_sub = ft.Text(self.t("play_empty_sub"))
        
        self.progress_bar = ft.ProgressBar(value=0.0, color=ft.Colors.PRIMARY, bgcolor=ft.Colors.GREY_300)
        self.lbl_time_current = ft.Text("00:00")
        self.lbl_time_total = ft.Text("00:00")
        self.lbl_status = ft.Text("", size=12, color=ft.Colors.GREY_700)
        self.btn_status_close = ft.IconButton(
            icon=ft.Icons.CLOSE,
            icon_size=16,
            visible=False,
            on_click=lambda e: self.on_status_close(e) if self.on_status_close else None,
            tooltip="Close",
        )

        status_row_controls: list[ft.Control] = [
            ft.Container(content=self.lbl_status, expand=True),
            self.btn_status_close,
        ]

        player_info_controls: list[ft.Control] = [
            ft.Row(
                controls=status_row_controls,
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ]

        # Connect slider change events to controller hooks.
        slider_jitter = ft.Slider(min=0.0, max=0.05, value=0.012, divisions=50, label=None)
        slider_jitter.on_change = lambda e: self.on_jitter_change(float(e.control.value)) if self.on_jitter_change and e.control.value is not None else None
        
        slider_stagger = ft.Slider(min=0.0, max=0.1, value=0.025, divisions=50, label=None)
        slider_stagger.on_change = lambda e: self.on_stagger_change(float(e.control.value)) if self.on_stagger_change and e.control.value is not None else None

        # Hesitation sliders.
        slider_hesitation_min = ft.Slider(min=0.01, max=0.12, value=0.03, divisions=110, label=None)
        slider_hesitation_min.on_change = lambda e: self.on_hesitation_min_change(float(e.control.value)) if self.on_hesitation_min_change and e.control.value is not None else None

        slider_hesitation_max = ft.Slider(min=0.01, max=0.12, value=0.05, divisions=110, label=None)
        slider_hesitation_max.on_change = lambda e: self.on_hesitation_max_change(float(e.control.value)) if self.on_hesitation_max_change and e.control.value is not None else None

        # Keybind dropdowns.
        keybind_options = [ft.dropdown.Option(label) for label in KEYBIND_MAP]

        dd_key_ctrl = ft.Dropdown(
            width=120,
            options=list(keybind_options),
            value="Ctrl",
            on_select=lambda e: self.on_keybind_change("ctrl", e.control.value) if self.on_keybind_change and e.control.value else None,
        )
        dd_key_shift = ft.Dropdown(
            width=120,
            options=list(keybind_options),
            value="Shift",
            on_select=lambda e: self.on_keybind_change("shift", e.control.value) if self.on_keybind_change and e.control.value else None,
        )

        # Connect play button to controller hook.
        self.btn_play = ft.FloatingActionButton(
            icon=ft.Icons.PLAY_ARROW,
            bgcolor=ft.Colors.PRIMARY,
            foreground_color=ft.Colors.WHITE,
            on_click=lambda e: self.on_play_click(e) if self.on_play_click else None
        )
        self.btn_mode_normal = ft.Button(
            self.t("play_mode_normal"),
            on_click=lambda e: self._handle_play_mode_button_click("normal"),
        )
        self.btn_mode_repeat_one = ft.Button(
            self.t("play_mode_repeat_one"),
            on_click=lambda e: self._handle_play_mode_button_click("repeat_one"),
        )
        self.btn_mode_repeat_all = ft.Button(
            self.t("play_mode_repeat_all"),
            on_click=lambda e: self._handle_play_mode_button_click("repeat_all"),
        )
        self.switch_split_enabled = ft.Switch(
            label=self.t("play_split_toggle"),
            value=self.split_enabled,
            on_change=lambda e: self._handle_split_toggle_change(bool(e.control.value)),
        )
        self._refresh_play_mode_buttons()
        self.split_role_row = ft.Row(
            controls=[ft.Text(self.t("play_split_label"), size=12, color=ft.Colors.GREY_600)],
            wrap=True,
            spacing=8,
            visible=False,
        )
        self._refresh_split_role_buttons()
        self.btn_prev = ft.IconButton(
            ft.Icons.SKIP_PREVIOUS,
            icon_size=30,
            tooltip=self.t("play_prev_unavailable"),
            disabled=True,
            on_click=lambda e: self.on_prev_click(e) if self.on_prev_click else None,
        )
        self.btn_next = ft.IconButton(
            ft.Icons.SKIP_NEXT,
            icon_size=30,
            tooltip=self.t("play_next_unavailable"),
            disabled=True,
            on_click=lambda e: self.on_next_click(e) if self.on_next_click else None,
        )

        tuning_controls: list[ft.Control] = [
            ft.Text(self.t("play_tuning"), size=16, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            self.create_slider_row(self.t("play_jitter"), self.t("play_jitter_desc"), slider_jitter),
            ft.Container(height=10),
            self.create_slider_row(self.t("play_stagger"), self.t("play_stagger_desc"), slider_stagger),
            ft.Container(height=10),
            ft.Text(self.t("play_hesitation"), size=14, weight=ft.FontWeight.BOLD),
            ft.Text(self.t("play_hesitation_desc"), size=12, color=ft.Colors.GREY_600),
            ft.Divider(),
            self.create_slider_row(self.t("play_hesitation_min"), self.t("play_hesitation_min_desc"), slider_hesitation_min),
            ft.Container(height=10),
            self.create_slider_row(self.t("play_hesitation_max"), self.t("play_hesitation_max_desc"), slider_hesitation_max),
        ]

        keybind_controls: list[ft.Control] = [
            ft.Text(self.t("play_keybinds"), size=16, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Row(
                controls=[
                    ft.Column([ft.Text(self.t("play_key_ctrl"), size=12), dd_key_ctrl], spacing=4),
                    ft.Column([ft.Text(self.t("play_key_shift"), size=12), dd_key_shift], spacing=4),
                ],
                wrap=True,
                spacing=16,
            ),
        ]

        progress_panel = ft.Card(
            content=ft.Container(
                padding=ft.padding.symmetric(horizontal=16, vertical=12),
                content=ft.Column(
                    spacing=8,
                    controls=[
                        ft.Row(
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            controls=[
                                ft.Text(self.t("play_progress_label"), size=12, color=ft.Colors.GREY_700),
                                ft.Row(
                                    spacing=8,
                                    controls=[
                                        self.lbl_time_current,
                                        ft.Text("/", color=ft.Colors.GREY_600),
                                        self.lbl_time_total,
                                    ],
                                ),
                            ],
                        ),
                        ft.Container(
                            border_radius=8,
                            bgcolor=ft.Colors.GREY_100,
                            padding=ft.padding.symmetric(horizontal=2, vertical=2),
                            content=self.progress_bar,
                        ),
                    ],
                ),
            )
        )

        return ft.Column(
            controls=[
                progress_panel,
                ft.Card(
                    content=ft.Container(
                        content=ft.Column(
                            controls=[
                                ft.ListTile(leading=ft.Icon(ft.Icons.AUDIOTRACK, color=ft.Colors.PRIMARY), title=self.lbl_track_title, subtitle=self.lbl_track_sub),
                                ft.Container(
                                    padding=ft.Padding.only(left=20, right=20, bottom=20),
                                    content=ft.Column(
                                        controls=[
                                            ft.Row(
                                                controls=[
                                                    ft.Text(self.t("play_mode_label"), size=12, color=ft.Colors.GREY_600),
                                                    self.btn_mode_normal,
                                                    self.btn_mode_repeat_one,
                                                    self.btn_mode_repeat_all,
                                                ],
                                                wrap=True,
                                                spacing=8,
                                            ),
                                            self.switch_split_enabled,
                                            self.split_role_row,
                                            *player_info_controls,
                                        ]
                                    )
                                )
                            ]
                        ),
                        padding=10
                    )
                ),
                ft.Card(content=ft.Container(content=ft.Column(controls=tuning_controls), padding=20)),
                ft.Card(content=ft.Container(content=ft.Column(controls=keybind_controls), padding=20)),
                ft.Row(controls=[self.btn_prev, self.btn_play, self.btn_next], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
            ], expand=True, spacing=20, scroll=ft.ScrollMode.AUTO
        )

    def build_library_view(self) -> ft.Column:
        """Build library view with search field and imported MIDI list."""
        self.library_search_field = ft.TextField(
            prefix_icon=ft.Icons.SEARCH,
            hint_text=self.t("lib_search"),
            border_radius=15,
            on_change=lambda e: self._refresh_library_list(),
        )
        self.library_list_view = ft.ListView(expand=True, spacing=10, controls=[])
        self.btn_library_select_all = ft.Button(
            self.t("lib_select_all"),
            icon=ft.Icons.SELECT_ALL,
            on_click=lambda e: self._handle_library_select_all_click(),
        )
        self.btn_library_invert = ft.Button(
            self.t("lib_invert_selection"),
            icon=ft.Icons.SWAP_HORIZ,
            on_click=lambda e: self._handle_library_invert_selection_click(),
        )
        self.btn_library_remove = ft.Button(
            self.t("lib_remove_selected"),
            icon=ft.Icons.DELETE_SWEEP,
            disabled=True,
            on_click=lambda e: self._handle_library_remove_selected_click(),
        )
        self._refresh_library_list()

        return ft.Column(
            controls=[
                ft.Text(self.t("lib_title"), size=24, weight=ft.FontWeight.BOLD),
                self.library_search_field,
                self.library_list_view,
                ft.Row(
                    controls=[
                        self.btn_library_select_all,
                        self.btn_library_invert,
                        self.btn_library_remove,
                        ft.FloatingActionButton(
                            icon=ft.Icons.ADD,
                            tooltip=self.t("lib_import"),
                            bgcolor=ft.Colors.PRIMARY,
                            foreground_color=ft.Colors.WHITE,
                            on_click=lambda e: self.on_import_click(e) if self.on_import_click else None,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
            ], expand=True, spacing=20
        )

    def build_settings_view(self) -> ft.Column:
        """Build settings view including language and display switches."""
        lang_display_map = {"en": "English", "ja": "日本語", "zh": "简体中文"}
        return ft.Column(
            controls=[
                ft.Text(self.t("set_title"), size=24, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.ListTile(
                    title=ft.Text(self.t("set_lang")), subtitle=ft.Text(self.t("set_lang_sub")),
                    trailing=ft.Dropdown(
                        width=150, options=[ft.dropdown.Option("English"), ft.dropdown.Option("日本語"), ft.dropdown.Option("简体中文")],
                        value=lang_display_map.get(self.current_lang, "English"),
                        on_select=self.change_language
                    )
                ),
                ft.Switch(label=self.t("set_dark"), value=self.page.theme_mode == ft.ThemeMode.DARK, on_change=self.toggle_theme),
                ft.Button(self.t("set_hotkey"), icon=ft.Icons.KEYBOARD),
                ft.Text(self.t("set_version", APP_VERSION), size=12, color=ft.Colors.GREY_500),
                ft.Text(self.t("set_project_meta"), size=12, color=ft.Colors.GREY_500)
            ], expand=True, spacing=15
        )