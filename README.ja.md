# StarResonanceMidi


MIDI をキーボード入力に変換する Flet GUI ツールです。プレイリスト連続再生に対応しています。

主な用途: MIDI ファイルをキーボード入力へ変換し、「星痕共鳴 / Blue Protocol: Star Resonance / ブループロトコル：スターレゾナンス」での演奏に利用すること。

**言語ページ：** 

[![English](https://img.shields.io/badge/English-README-blue?style=flat-square)](README.md)
[![Chinese](https://img.shields.io/badge/中文-README-red?style=flat-square)](README.zh-CN.md)
[![Japanese](https://img.shields.io/badge/日本語-README-lightgrey?style=flat-square)](README.ja.md)

## ライセンス

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-orange?style=flat-square&logo=gnu)](https://www.gnu.org/licenses/agpl-3.0)


## 楽曲/MIDI の著作権および商標・権利表記

- 本ソフトウェアは非公式の独立ツールであり、ゲーム運営・権利者との提携や公認はありません。
- 利用する楽曲および MIDI データの適法性は利用者の責任で確認してください。
- 権利許諾のない楽曲/MIDI の配布・公開・演奏は行わないでください。
- 商標/名称「Blue Protocol」は **BANDAI NAMCO** に帰属します。
- 「星痕共鳴 / Blue Protocol: Star Resonance / ブループロトコル：スターレゾナンス」の著作権は **BOKURA** に帰属します。


## 必要環境

- Python 3.12+
- 依存パッケージ：`flet`、`mido`、`pynput`

インストール例：

```bash
python -m pip install flet mido pynput
```

## 実行

プロジェクトルートで実行：

```bash
python main.py
```


## パッケージング

Windows のンインストーラーは、PyAppify + GitHub Actions で生成しています。


## 翻訳コントリビューション

他言語への翻訳コントリビューションを歓迎します。

[![Help Translate](https://img.shields.io/badge/Translation-Help_Translate-brightgreen?style=flat-square&logo=google-translate)](CONTRIBUTING.md)