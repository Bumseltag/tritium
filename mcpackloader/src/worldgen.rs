use std::{fs::File, io::BufReader};

#[cfg(feature = "more_const")]
use bevy_platform::collections::hash_map::HashMap;
#[cfg(not(feature = "more_const"))]
use hashbrown::HashMap;
use serde::Deserialize;

use crate::{ResourceLocation, ResourceType};

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

/// A density function
#[derive(Deserialize)]
#[serde(untagged)]
pub enum DensityFunction {
    Value(f64),
    File(ResourceLocation<DensityFunction>),
    Function {
        ty: String,
        params: HashMap<String, DensityFunction>,
    },
}

impl ResourceType for DensityFunction {
    fn to_path(res_loc: &ResourceLocation<Self>) -> std::path::PathBuf {
        res_loc.to_data_path("worldgen/density_function", "json")
    }

    fn open(path: std::path::PathBuf) -> Result<Self, crate::ResourceParseError> {
        Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
    }
}

#[derive(Deserialize)]
pub struct DensityFunctionOp {
    ty: String,
    #[serde(flatten)]
    params: HashMap<String, DensityFunction>,
}

// impl<'de> Deserialize<'de> for DensityFunction {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: serde::Deserializer<'de>,
//     {
//         deserializer.deserialize_any(DensityFunctionVisitor)
//     }
// }

// #[derive(Debug)]
// struct DensityFunctionVisitor;

// impl<'de> Visitor<'de> for DensityFunctionVisitor {
//     type Value = DensityFunction;
//     fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
//         formatter.write_str("a float, resource location or density function")
//     }

//     fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
//     where
//         E: serde::de::Error,
//     {
//         Ok(DensityFunction::Value(v))
//     }

//     fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
//     where
//         E: serde::de::Error,
//     {
//         Ok(DensityFunction::File(
//             ResourceLocation::from_str(v).unwrap(),
//         ))
//     }

//     fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
//     where
//         A: serde::de::MapAccess<'de>,
//     {
//         let mut params = if let Some(size) = map.size_hint() {
//             HashMap::with_capacity(size)
//         } else {
//             HashMap::new()
//         };
//         let mut ty = None;
//         while let Some((key, value)) = map.next_entry()? {
//             if key == "type" {
//                 ty = Some(value);
//             } else {
//                 params.insert(key, value);
//             }
//         }

//         Ok(DensityFunction::Function { ty: ty.unwrap(), params })
//     }
// }

#[cfg(test)]
mod tests {
    #[test]
    fn parse_density_function() {}
}
