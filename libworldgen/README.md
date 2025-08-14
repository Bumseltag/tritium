# libworldgen

libworldgen a Rust libary for Minecraft world generation, and the heart of tritium.

## Testing

Certain functions in libworldgen are tested by comparing them to their Java equivalents in Minecraft.
This is done using the `libworldgen-java-tests` gradle project, which generates a jar containing the minecraft code with the `uberJar` task.
The jar is then loaded by the Rust code and the respective functions are ran using [jni](https://crates.io/crates/jni).

To run the tests, do:

```sh
cargo test -F java_tests
```

This will first run the `uberJar` gradle task to compile everything,
then run the tests. Note that the gradle task can take a minute the first time it runs,
all subsequent runs should be very fast however.

You can opt out of all the java tests by disabling the `java_tests` feature flag:

```sh
cargo test
```
