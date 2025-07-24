# Tritium

> Hardware-Accelerated World Generation for Minecraft

This mod improves minecraft world generation *without* changing vanilla mechanics,
using a special GPU-powered algorithm written in Rust.

This repo consists of multiple sub-projects:

- `libworldgen`: the main world generation algorithm, written in Rust
- `tritium`: the actual minecraft mod, powered by libworldgen
- `worldgenpreview`: a spectator-mode-like viewer for world generation, so you don't have to boot up the game for every little change
- `mcpackloader`: a datapack loader for `libworldgen` and `worldgenpreview`

## Development

### Rust

This repo requires no special setup for Rust. If you haven't installed it yet, do the following steps:

1. Install rust with [rustup](https://rustup.rs/). Choose default installation settings.
1. If you're using VSCode, install the `rust-analyzer` plugin.
1. If you're using `rust-analyzer`, you will need to open the sub-project you want to work on in a seperate window.
1. Follow the steps the sub-project's README.

### Profiling

The sub-projects in the repo support profiling with Tracy.

To begin profiling, install [Tracy 0.12.2](https://github.com/wolfpld/tracy/releases/tag/v0.12.2). Run `tracy-capture.exe -fo log.tracy`, then start the application to be profiled, once done, close the application and `tracy-capture` will stop automatically. Now you can open the created `log.tracy` file with `tracy-profiler.exe`.