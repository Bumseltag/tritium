use std::{fs::File, io::BufReader};

use serde::Deserialize;
use serde_json::Value;

use crate::{ResourceLocation, ResourceType};

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct NoiseParameters {
    pub amplitudes: Vec<f64>,
    #[serde(rename = "firstOctave")]
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

/// A density function.
///
/// This is just a wrapper around a json [`Value`].
#[derive(Deserialize, Debug, Clone, PartialEq, Hash)]
pub struct DensityFunction(pub Value);

impl ResourceType for DensityFunction {
    fn to_path(res_loc: &ResourceLocation<Self>) -> std::path::PathBuf {
        res_loc.to_data_path("worldgen/density_function", "json")
    }

    fn open(path: std::path::PathBuf) -> Result<Self, crate::ResourceParseError> {
        Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
    }
}
