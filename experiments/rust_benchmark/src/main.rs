use chd_rust_benchmark::{Corpus, Detection, Tracker};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use serde_json::json;
use std::{error::Error, fs, hint::black_box, time::Instant};

fn peak_rss_kib() -> i64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    unsafe {
        libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr());
        usage.assume_init().ru_maxrss
    }
}

fn replay(args: &[String]) -> Result<(), Box<dyn Error>> {
    let corpus: Corpus = serde_json::from_slice(&fs::read(&args[2])?)?;
    let case = corpus
        .cases
        .iter()
        .find(|case| case.name == args[3])
        .ok_or("case not found")?;
    let repetitions: usize = args[4].parse()?;
    let mut warm = Tracker::default();
    for frame in &case.frames {
        black_box(warm.process(frame));
    }
    let mut latencies = Vec::with_capacity(case.frames.len() * repetitions);
    let mut elapsed = 0.0;
    for _ in 0..repetitions {
        let mut tracker = Tracker::default();
        let start_pass = Instant::now();
        for frame in &case.frames {
            let start = Instant::now();
            black_box(tracker.process(frame));
            latencies.push(start.elapsed().as_secs_f64() * 1e6);
        }
        elapsed += start_pass.elapsed().as_secs_f64();
    }
    if let Some(path) = args.get(5) {
        let mut tracker = Tracker::default();
        let trace: Vec<_> = case
            .frames
            .iter()
            .map(|frame| tracker.process(frame))
            .collect();
        fs::write(path, serde_json::to_vec(&trace)?)?;
    }
    println!(
        "{}",
        json!({"language":"rust","case":case.name,"kind":case.kind,"elapsed_seconds":elapsed,"frames":latencies.len(),"latencies_us":latencies,"peak_rss_kib":peak_rss_kib()})
    );
    Ok(())
}

fn infer(args: &[String]) -> Result<(), Box<dyn Error>> {
    let library = std::env::var("ORT_DYLIB_PATH")?;
    let environment = ort::init_from(library)?
        .with_name("chd-comparison")
        .build()?;
    let mut builder = Session::builder(&environment)?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .with_parallel_execution(false)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?;
    if args[5] == "cuda" {
        builder = builder.with_execution_providers([ort::ep::CUDA::default()
            .with_memory_limit(1 << 30)
            .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
            .with_conv_max_workspace(false)
            .build()
            .error_on_failure()])?;
    }
    let mut session = builder.commit_from_file(&args[2])?;
    let raw = fs::read(&args[3])?;
    let count: usize = args[4].parse()?;
    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let stride = 3 * 640 * 640;
    if values.len() != count * stride {
        return Err("invalid input buffer length".into());
    }
    let inputs: Vec<_> = values
        .chunks_exact(stride)
        .map(|frame| TensorRef::from_array_view((vec![1usize, 3, 640, 640], frame)))
        .collect::<Result<_, _>>()?;
    let runs: usize = args[6].parse()?;
    let warmup: usize = args[7].parse()?;
    let post = args[8] == "post";
    for i in 0..warmup {
        let output = session.run(ort::inputs![&*inputs[i % count]])?;
        black_box(output[0].try_extract_tensor::<f32>()?.1[4]);
    }
    let mut tracker = Tracker::default();
    let mut latencies = Vec::with_capacity(runs);
    let all = Instant::now();
    for i in 0..runs {
        let start = Instant::now();
        let output = session.run(ort::inputs![&*inputs[i % count]])?;
        let (_, data) = output[0].try_extract_tensor::<f32>()?;
        if post {
            let detections: Vec<Detection> = data
                .chunks_exact(6)
                .filter(|row| row[4] >= 0.25)
                .map(|row| {
                    [
                        row[0] as f64,
                        row[1] as f64,
                        row[2] as f64,
                        row[3] as f64,
                        row[4] as f64,
                        row[5] as f64,
                    ]
                })
                .collect();
            black_box(tracker.process(&detections));
        } else {
            black_box(data[4]);
        }
        latencies.push(start.elapsed().as_secs_f64() * 1e6);
    }
    let elapsed = all.elapsed().as_secs_f64();
    if let Some(path) = args.get(9) {
        let mut raw_output = Vec::new();
        for input in &inputs {
            let output = session.run(ort::inputs![&**input])?;
            let (_, data) = output[0].try_extract_tensor::<f32>()?;
            for value in data {
                raw_output.extend_from_slice(&value.to_le_bytes());
            }
        }
        fs::write(path, raw_output)?;
    }
    println!(
        "{}",
        json!({"language":"rust","backend":args[5],"post":post,"elapsed_seconds":elapsed,"frames":runs,"latencies_us":latencies,"peak_rss_kib":peak_rss_kib()})
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("replay") => replay(&args),
        Some("infer") => infer(&args),
        _ => Err("usage: replay CORPUS CASE REPS [TRACE]; infer MODEL INPUT COUNT cpu|cuda RUNS WARMUP post|raw [OUTPUT]".into()),
    }
}
