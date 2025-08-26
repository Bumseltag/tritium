use mcpackloader::{ResourceLocation, ResourceParseError};

use crate::density_function::DynFnOpType;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Found unknown density function operation: {0}")]
    UnknownDensityFunctionOp(ResourceLocation<DynFnOpType>),
    #[error("Error while parsing json: {0}")]
    JsonParseError(String),
    #[error("Error while parsing json: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Error while loading resource: {0}")]
    ResourceLoadError(#[from] ResourceParseError),
}
