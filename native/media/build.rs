use std::{env, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-changed=media.c");
    println!("cargo:rerun-if-env-changed=FFMPEG_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_INCLUDE");
    println!("cargo:rerun-if-env-changed=CUDA_LIB");
    let mut build = cc::Build::new();
    build.file("media.c").std("c11").warnings(true);
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        build.define("CHD_CUDA", None);
        build.include(env::var("CUDA_INCLUDE").expect("CUDA_INCLUDE is required for NVDEC"));
        if let Ok(path) = env::var("CUDA_LIB") {
            println!("cargo:rustc-link-search=native={path}");
        }
    }
    let explicit_root = env::var_os("FFMPEG_ROOT");
    if let Some(root) = &explicit_root {
        let root = PathBuf::from(root);
        build
            .include(root.join("usr/include/x86_64-linux-gnu"))
            .include(root.join("usr/include"));
        println!(
            "cargo:rustc-link-search=native={}/usr/lib/x86_64-linux-gnu",
            root.display()
        );
    } else {
        for lib in ["libavformat", "libavcodec", "libavutil", "libswscale"] {
            let found = pkg_config::Config::new()
                .cargo_metadata(false)
                .probe(lib)
                .expect("Install FFmpeg development packages or set FFMPEG_ROOT");
            for path in found.include_paths {
                build.include(path);
            }
        }
    }
    build.compile("construction_media");
    // --as-needed linkers must see the static bridge before its shared libs,
    // including when this crate is tested directly rather than through a user.
    if explicit_root.is_some() {
        for lib in ["avformat", "avcodec", "avutil", "swscale"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    } else {
        for lib in ["libavformat", "libavcodec", "libavutil", "libswscale"] {
            pkg_config::Config::new()
                .probe(lib)
                .expect("FFmpeg link metadata");
        }
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        println!("cargo:rustc-link-lib=cuda");
    }
}
