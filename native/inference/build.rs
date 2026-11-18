use std::{env, path::PathBuf};

fn main() {
    for name in [
        "TRT_INCLUDE",
        "TRT_LIB",
        "CUDA_INCLUDE",
        "CUDA_LIB",
        "OPENCV_ROOT",
        "FFMPEG_ROOT",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for file in [
        "bridge.cpp",
        "preprocess.cu",
        "gpu_frames.inc",
        "gpu_api.inc",
        "nv12_api.inc",
        "gpu_encoder.inc",
    ] {
        println!("cargo:rerun-if-changed={file}");
    }
    if env::var_os("CARGO_FEATURE_TENSORRT").is_none() {
        return;
    }
    let trt = env::var("TRT_INCLUDE").expect("TRT_INCLUDE must point to matching TensorRT headers");
    let cuda =
        env::var("CUDA_INCLUDE").expect("CUDA_INCLUDE must point to complete CUDA toolkit headers");
    let cv = env::var("OPENCV_ROOT").unwrap_or_else(|_| "/".into());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel = std::fs::read_to_string("preprocess.cu").unwrap();
    std::fs::write(
        out.join("preprocess_source.h"),
        format!("static const char* kernel_source=R\"KERNEL({kernel})KERNEL\";"),
    )
    .unwrap();
    let mut bridge = cc::Build::new();
    if env::var_os("CARGO_FEATURE_GPU_MEDIA").is_some() {
        let root = env::var("FFMPEG_ROOT").unwrap_or_else(|_| "/".into());
        bridge
            .define("CHD_GPU_MEDIA", None)
            .include(format!("{root}/usr/include/x86_64-linux-gnu"));
        println!("cargo:rustc-link-search=native={root}/usr/lib/x86_64-linux-gnu");
    }
    bridge
        .cpp(true)
        .std("c++17")
        .file("bridge.cpp")
        .include(trt)
        .include(cuda)
        .include(out)
        .include(format!("{cv}/usr/include/opencv4"))
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("construction_trt_bridge");
    if env::var_os("CARGO_FEATURE_GPU_MEDIA").is_some() {
        println!("cargo:rustc-link-lib=avcodec");
        println!("cargo:rustc-link-lib=avutil");
    }
    for name in ["TRT_LIB", "CUDA_LIB"] {
        if let Ok(path) = env::var(name) {
            println!("cargo:rustc-link-search=native={path}");
        }
    }
    println!("cargo:rustc-link-search=native={cv}/usr/lib/x86_64-linux-gnu");
    for lib in [
        "nvinfer",
        "nvinfer_plugin",
        "cudart",
        "nvrtc",
        "cuda",
        "opencv_core",
        "opencv_imgproc",
    ] {
        println!("cargo:rustc-link-lib={lib}");
    }
}
