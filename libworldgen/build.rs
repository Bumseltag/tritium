use std::{
    env,
    fs::{self, File},
    path::PathBuf,
    process::{Command, Stdio},
    str::FromStr,
};

use walkdir::WalkDir;

fn main() {
    if feature("compile_java") {
        compile_java();
    }
}

fn compile_java() {
    let java_code_changed = has_changed(
        &[
            "libworldgen-java-tests/settings.gradle.kts",
            "libworldgen-java-tests/gradle.properties",
            "libworldgen-java-tests/app/build.gradle.kts",
            "libworldgen-java-tests/app/src",
        ],
        "libworldgen-java-tests-last-changed",
    );
    if !java_code_changed {
        return;
    }

    let gradlew = if cfg!(windows) {
        "./libworldgen-java-tests/gradlew.bat"
    } else {
        "./libworldgen-java-tests/gradlew"
    };
    let output = Command::new(gradlew)
        .args(["--no-daemon", "--console=plain", "uberJar"])
        .stdout(Stdio::piped())
        .current_dir(
            PathBuf::from_str("libworldgen-java-tests")
                .unwrap()
                .canonicalize()
                .unwrap(),
        )
        .output()
        .unwrap();
    if !output.status.success() {
        println!(
            "cargo::error=Gradle failed to compile libworldgen-java-tests (error code: {:?})",
            output.status.code()
        );
        println!("cargo::error=");
        println!("cargo::error=stdout:");
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            println!("cargo::error={line}");
        }
    }

    save_last_changed("libworldgen-java-tests-last-changed");
}

fn has_changed(watch: &[&str], id: &str) -> bool {
    let last_change_file = PathBuf::from_str(&("target".to_owned() + "/" + id)).unwrap();
    let last_run = if fs::exists(&last_change_file).unwrap() {
        fs::metadata(last_change_file).unwrap().modified().unwrap()
    } else {
        return true;
    };
    for path in watch {
        let meta = fs::metadata(path).unwrap();
        if meta.is_file() && meta.modified().unwrap() > last_run {
            return true;
        } else if meta.is_dir() {
            for entry in WalkDir::new("foo").into_iter().filter_map(|e| e.ok()) {
                let last_modified = entry.metadata().unwrap().modified().unwrap();
                if last_modified > last_run {
                    return true;
                }
            }
        }
    }
    false
}

fn save_last_changed(id: &str) {
    File::create("target".to_owned() + "/" + id).unwrap();
}

fn feature(f: &str) -> bool {
    env::var("CARGO_CFG_FEATURE")
        .unwrap()
        .split(",")
        .any(|v| v == f)
}
