val minecraftVersion = providers.gradleProperty("minecraft_version").get()
val forgeVersion = providers.gradleProperty("forge_version").get()

plugins {
    `java-library`

    id("net.minecraftforge.gradle").version("[6.0.36,6.2)")
}

repositories {
    // Use Maven Central for resolving dependencies.
    mavenCentral()
    maven {
        name = "Forge"
        url = uri("https://maven.minecraftforge.net")
    }
    maven {
        name = "Minecraft Libraries"
        url = uri("https://libraries.minecraft.net")
    }
    exclusiveContent {
        forRepository {
            maven {
                name = "Sponge"
                url =  uri("https://repo.spongepowered.org/repository/maven-public")
            }
        }
        filter {
            includeGroupAndSubgroups("org.spongepowered")
        }
    }
}

minecraft {
    mappings(mapOf(
        "channel" to "official",
        "version" to minecraftVersion
    ))

    reobf = false
}

dependencies {
//    implementation(libs.guava)

    minecraft("net.minecraftforge:forge:${minecraftVersion}-${forgeVersion}")
    implementation("org.slf4j:slf4j-simple:2.0.9")
}

// Apply a specific Java toolchain to ease working on different environments.
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(17)
    }
}

tasks.register<Jar>("uberJar") {
    archiveClassifier = "uber"
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE

    from(sourceSets.main.get().output)

    dependsOn(configurations.runtimeClasspath)
    from({
        configurations.runtimeClasspath.get().filter {
            it.name.endsWith("jar")
                    && !it.name.equals("client-extra.jar")
                    && !it.name.equals("slf4j-simple-1.7.30.jar")
        }.map {
            jar -> zipTree(jar).matching {
                include("**/*.class")
                include("**/*.SLF4JServiceProvider")
            }
        }
    })
}