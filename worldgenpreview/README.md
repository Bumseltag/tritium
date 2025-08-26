# worldgenpreview

Worldgenpreview is a small front-end for libworldgen written using the [Bevy Engine](https://bevy.org/).

To run it, you must add at least 1 resource pack. This should most likely be the `client.jar` of the version you are using (download it from [the wiki](https://minecraft.wiki/w/1.21)).
You can also add your own resource packs (directories or zip files), and mods.

```sh
cargo run -- resources path/to/client.jar
```

Here we're adding the folder "resources" as a resource pack, which contains data for development and testing.
Ordering is important here. The same way you can choose the priority of a resource pack in the minecraft pack menu,
resource packs that are defined earlier will override later ones. That means that you should always put the `client.jar` last.
