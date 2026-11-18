use std::env;
use std::path::PathBuf;

fn main() {
    let trt_include = env::var("TRT_INCLUDE").expect("Set TRT_INCLUDE to TensorRT headers");
    let trt_lib = env::var("TRT_LIB").expect("Set TRT_LIB to matching TensorRT libraries");
    let cuda_include = env::var("CUDA_INCLUDE").expect("Set CUDA_INCLUDE");
    let cuda_lib = env::var("CUDA_LIB").expect("Set CUDA_LIB");
    let cv_root = env::var("OPENCV_ROOT").expect("Set OPENCV_ROOT");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let source = std::fs::read_to_string("preprocess.cu").unwrap();
    std::fs::write(
        out.join("preprocess_source.h"),
        format!("static const char* kernel_source = R\"KERNEL({source})KERNEL\";"),
    )
    .unwrap();
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("bridge.cpp")
        .include(trt_include)
        .include(cuda_include)
        .include(out)
        .include(format!("{cv_root}/usr/include/opencv4"))
        .compile("chd_trt_bridge");
    for path in [
        &trt_lib,
        &cuda_lib,
        &format!("{cv_root}/usr/lib/x86_64-linux-gnu"),
    ] {
        println!("cargo:rustc-link-search=native={path}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{path}");
    }
    for library in [
        "nvinfer",
        "nvinfer_plugin",
        "cudart",
        "nvrtc",
        "cuda",
        "opencv_core",
        "opencv_imgproc",
    ] {
        println!("cargo:rustc-link-lib={library}");
    }
    for name in [
        "TRT_INCLUDE",
        "TRT_LIB",
        "CUDA_INCLUDE",
        "CUDA_LIB",
        "OPENCV_ROOT",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!("cargo:rerun-if-changed=bridge.cpp");
    println!("cargo:rerun-if-changed=preprocess.cu");
}
