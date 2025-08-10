# worldgenpreview

Worldgenpreview is a small front-end for libworldgen written using the [Bevy Engine](https://bevy.org/).

To run it, you must add at least 1 resource pack. This should most likely be the `client.jar` of the version you are using (download it from [the wiki](https://minecraft.wiki/w/1.21)).
You can also add your own resource packs (directories or zip files), and mods.

```sh
cargo run -- path/to/my/resourcepack path/to/client.jar
```

Note that ordering is important. Earlier resource packs will override later resource packs. That means that you will have to put your `client.jar` last.
