# worldgenpreview

Worldgenpreview is a small front-end for libworldgen written using the [Bevy Engine](https://bevy.org/).

Run it with
```sh
cargo run -- --resources path/to/minecraft/jar
```

> [!NOTE]
> `--resources` is not yet supported (issue [#6](https://github.com/Bumseltag/tritium/issues/6)). The following is how it's gonna look like once completed.

If you pass in a .jar file into `--resources`, it will extract that file into `./resources`. After the first time you run it, you will no longer have to specify `--resources`, as it will default to that folder.