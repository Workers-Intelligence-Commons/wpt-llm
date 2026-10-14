//! Records which compiler built the extension, so dataset builds can report
//! it next to the reference version in rust-toolchain.toml.

use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let output = Command::new(rustc).arg("--version").output().expect("rustc --version");
    let version = String::from_utf8(output.stdout).expect("rustc --version prints UTF-8");
    println!("cargo:rustc-env=WPT_RUSTC_VERSION={}", version.trim());
    println!("cargo:rerun-if-changed=build.rs");
}
