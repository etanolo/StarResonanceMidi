# StarResonanceMidi


这是一个将 MIDI 映射为键盘输入的 Flet 图形工具，支持播放列表顺序播放。

主要用途：将 MIDI 文件映射到键盘，用于「星痕共鸣 / Blue Protocol: Star Resonance / ブループロトコル：スターレゾナンス」内演奏。

**语言版本：** 

[![English](https://img.shields.io/badge/English-README-blue?style=flat-square)](README.md)
[![Chinese](https://img.shields.io/badge/中文-README-red?style=flat-square)](README.zh-CN.md)
[![Japanese](https://img.shields.io/badge/日本語-README-lightgrey?style=flat-square)](README.ja.md)

## 许可证

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-orange?style=flat-square&logo=gnu)](https://www.gnu.org/licenses/agpl-3.0)


## 歌曲/MIDI 版权与商标版权声明

- 本软件为独立的非官方工具，与游戏发行方或权利方无隶属或背书关系。
- 你需要自行确保所使用歌曲与 MIDI 文件的合法性。
- 未经授权，请勿上传、传播或公开演奏受版权保护的歌曲/MIDI。
- 游戏商标/名称「蓝色协议 / Blue Protocol」归属于 **BANDAI NAMCO**。
- 游戏「星痕共鸣 / Blue Protocol: Star Resonance / ブループロトコル：スターレゾナンス」版权归属于 **BOKURA**。


## 环境要求

- Python 3.12+
- 依赖包：`flet`、`mido`、`pynput`

安装示例：

```bash
python -m pip install flet mido pynput
```

## 运行方式

在项目根目录执行：

```bash
python main.py
```


## 打包说明

Windows 安装包由 PyAppify + GitHub Actions 生成。


## 翻译贡献说明

欢迎提交其他语言版本的翻译。

[![Help Translate](https://img.shields.io/badge/Translation-Help_Translate-brightgreen?style=flat-square&logo=google-translate)](CONTRIBUTING.md)