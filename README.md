# StarResonanceMidi


StarResonanceMidi is a MIDI-to-keyboard playback tool with a Flet GUI and playlist sequencing.

Primary use case: map MIDI files to keyboard input for in-game performance in Star Resonance.

**Language Versions:** 

[![English](https://img.shields.io/badge/English-README-blue?style=flat-square)](README.md)
[![Chinese](https://img.shields.io/badge/中文-README-red?style=flat-square)](README.zh-CN.md)
[![Japanese](https://img.shields.io/badge/日本語-README-lightgrey?style=flat-square)](README.ja.md)


## License

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-orange?style=flat-square&logo=gnu)](https://www.gnu.org/licenses/agpl-3.0)


## Copyright, Music, and Trademark Notice

- This software is an independent fan-made tool and is not affiliated with or endorsed by game publishers or rights holders.
- You are responsible for the legality of any songs and MIDI files you use.
- Do not upload, distribute, or perform copyrighted songs/MIDI files without proper authorization.
- The trademark/name "Blue Protocol" belongs to **BANDAI NAMCO**.
- The game/copyright for "Star Resonance / Blue Protocol: Star Resonance / ブループロトコル：スターレゾナンス" belongs to **BOKURA**.


## Requirements

- Python 3.12+
- Packages used by this project: `flet`, `mido`, `pynput`, `music21`

Example installation:

```bash
python -m pip install flet mido pynput music21
```

## Run

From project root:

```bash
python main.py
```


## Packaging

Windows installer builds are produced with [Pyappify](https://github.com/ok-oldking/pyappify) via GitHub Actions.


## Translation Contributions

Contributions for additional languages are welcome.

[![Help Translate](https://img.shields.io/badge/Translation-Help_Translate-brightgreen?style=flat-square&logo=google-translate)](CONTRIBUTING.md)