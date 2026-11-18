//! JSONL parity/replay interface: no GPU, camera, database or external effects.
use crate::{
    config::{Rules, TrackerKind},
    hazard::{HazardEngine, Track},
    inference::Detection,
    tracking::Tracker,
};
use anyhow::Result;
use serde::Deserialize;
use std::{
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    #[serde(default)]
    detections: Vec<Detection>,
    #[serde(default)]
    tracked: Option<Vec<Track>>,
    #[serde(default)]
    reset: bool,
    #[serde(default)]
    rules: Rules,
    #[serde(default)]
    tracker: TrackerKind,
}
pub fn run(input: &Path, output: &Path) -> Result<()> {
    anyhow::ensure!(
        !output.exists() || std::fs::canonicalize(input)? != std::fs::canonicalize(output)?,
        "Input and output must differ"
    );
    let mut writer = BufWriter::new(std::fs::File::create(output)?);
    let mut tracker = Tracker::new(TrackerKind::Centroid);
    let mut hazard = HazardEngine::default();
    for line in BufReader::new(std::fs::File::open(input)?).lines() {
        let input: Input = serde_json::from_str(&line?)?;
        if input.reset {
            tracker = Tracker::new(input.tracker);
            hazard = HazardEngine::default();
        }
        let tracks = input
            .tracked
            .unwrap_or_else(|| tracker.update(&input.detections));
        let result = hazard.detect(&tracks, &input.rules)?;
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({"tracks":tracks,"warnings":result.warnings,"cone_polys":result.cone_polys,"pole_polys":result.pole_polys}),
        )?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}
