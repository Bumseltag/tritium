# Tritium

> Hardware-Accelerated World Generation for Minecraft

This mod improves minecraft world generation *without* changing vanilla mechanics,
using a special GPU-powered algorithm written in Rust.

This repo consists of multiple sub-projects:

- `libworldgen`: the main world generation algorithm, written in Rust (Status: ❌)
- `tritium`: the actual minecraft mod, powered by libworldgen (Status: ❌)
- `worldgenpreview`: a spectator-mode-like viewer for world generation, so you don't have to boot up the game for every little change (Status: ✅)
- `mcpackloader`: a datapack loader for `libworldgen` and `worldgenpreview` (Status: 🚧)
