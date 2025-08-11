use std::{fs::File, io::BufReader};

use serde::Deserialize;

use crate::ResourceType;

#[derive(Deserialize)]
pub struct NoiseParameters {
    pub amplitudes: Vec<f64>,
    pub first_octave: i32,
}

impl ResourceType for NoiseParameters {
    fn to_path(res_loc: &crate::ResourceLocation<Self>) -> std::path::PathBuf {
        res_loc.to_data_path("worldgen/noise", "json")
    }

    fn open(path: std::path::PathBuf) -> Result<Self, crate::ResourceParseError> {
        Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
    }
}
