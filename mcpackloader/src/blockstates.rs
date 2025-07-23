use std::{fs::File, io::BufReader, path::PathBuf};

#[cfg(feature = "more_const")]
use bevy_platform::collections::hash_map::HashMap;
#[cfg(not(feature = "more_const"))]
use hashbrown::HashMap;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{ResourceLocation, ResourceParseError, ResourceType, models::BlockModel};

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Blockstate(pub HashMap<String, Property>);

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Property {
    Int(i32),
    Bool(bool),
    Enum(String),
}

impl Property {
    pub fn parse(input: &str) -> Self {
        if let Ok(num) = input.parse() {
            Property::Int(num)
        } else if let Ok(boolean) = input.parse() {
            Property::Bool(boolean)
        } else {
            Property::Enum(input.to_owned())
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlockstateFile {
    Variants(Vec<(Vec<VariantBlockstate>, Vec<ConfiguredModel>)>),
    Multipart(Vec<(MultipartPredicate, Vec<ConfiguredModel>)>),
}

impl BlockstateFile {
    pub fn parse(json: Map<String, Value>) -> Result<Self, ResourceParseError> {
        if let Some(variants) = json.get("variants") {
            let Value::Object(variants) = variants else {
                return Err(ResourceParseError::InvalidType(variants.clone()));
            };

            let mut res_variants = vec![];

            for (blockstate, variant) in variants {
                let blockstate = VariantBlockstate::parse(blockstate)?;
                let models = ConfiguredModel::parse_model_pool(variant)?;
                res_variants.push((blockstate, models));
            }

            Ok(BlockstateFile::Variants(res_variants))
        } else if let Some(multipart) = json.get("multipart") {
            let multipart = multipart
                .as_array()
                .ok_or_else(|| ResourceParseError::InvalidType(multipart.clone()))?;

            Ok(BlockstateFile::Multipart(
                multipart
                    .iter()
                    .map(|part| {
                        let part = part
                            .as_object()
                            .ok_or_else(|| ResourceParseError::InvalidType(part.clone()))?;
                        let predicate = part.get("key");
                        let predicate = if let Some(predicate) = predicate {
                            MultipartPredicate::parse(predicate.as_object().ok_or_else(|| {
                                ResourceParseError::InvalidType(predicate.clone())
                            })?)?
                        } else {
                            MultipartPredicate::Always
                        };
                        let models = part.get("apply").ok_or(ResourceParseError::NoApplyClause)?;
                        let models = ConfiguredModel::parse_model_pool(models)?;
                        Ok((predicate, models))
                    })
                    .collect::<Result<_, ResourceParseError>>()?,
            ))
        } else {
            Err(ResourceParseError::InvalidFormat(
                "Expected `variants` or `multipart`",
            ))
        }
    }
}

impl ResourceType for BlockstateFile {
    fn to_path(res_loc: &ResourceLocation<Self>) -> PathBuf {
        res_loc.to_assets_path("blockstates", "json")
    }

    fn open(path: PathBuf) -> Result<Self, ResourceParseError> {
        let json: Map<String, Value> = serde_json::from_reader(BufReader::new(File::open(path)?))?;
        BlockstateFile::parse(json)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantBlockstate {
    pub name: String,
    pub value: Property,
}

impl VariantBlockstate {
    pub fn parse(input: &str) -> Result<Vec<Self>, ResourceParseError> {
        let mut res = vec![];
        if input.is_empty() {
            return Ok(res);
        }
        for blockstate in input.split(",") {
            let Some((prop, value)) = blockstate.split_once('=') else {
                return Err(ResourceParseError::InvalidVariantBlockstates(
                    input.to_string(),
                ));
            };

            let value = Property::parse(value);
            res.push(VariantBlockstate {
                name: prop.to_owned(),
                value,
            });
        }
        Ok(res)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultipartPredicate {
    Always,
    Single(Vec<MultipartBlockstate>),
    And(Vec<Vec<MultipartBlockstate>>),
    Or(Vec<Vec<MultipartBlockstate>>),
}

impl MultipartPredicate {
    pub fn parse(input: &Map<String, Value>) -> Result<Self, ResourceParseError> {
        let (first_name, value) = input
            .iter()
            .next()
            .ok_or(ResourceParseError::EmptyMultipartWhen)?;
        match first_name.as_str() {
            "AND" => {
                let Some(blockstates) = value.as_array() else {
                    return Err(ResourceParseError::InvalidType(value.to_owned()));
                };

                Ok(MultipartPredicate::And(
                    blockstates
                        .iter()
                        .map(|blockstate| {
                            MultipartBlockstate::parse(blockstate.as_object().ok_or_else(|| {
                                ResourceParseError::InvalidType(blockstate.clone())
                            })?)
                        })
                        .collect::<Result<_, _>>()?,
                ))
            }
            "OR" => {
                let Some(blockstates) = value.as_array() else {
                    return Err(ResourceParseError::InvalidType(value.to_owned()));
                };

                Ok(MultipartPredicate::Or(
                    blockstates
                        .iter()
                        .map(|blockstate| {
                            MultipartBlockstate::parse(blockstate.as_object().ok_or_else(|| {
                                ResourceParseError::InvalidType(blockstate.clone())
                            })?)
                        })
                        .collect::<Result<_, _>>()?,
                ))
            }
            _ => Ok(MultipartPredicate::Single(MultipartBlockstate::parse(
                input,
            )?)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultipartBlockstate {
    pub name: String,
    pub values: Vec<Property>,
}

impl MultipartBlockstate {
    pub fn parse(obj: &Map<String, Value>) -> Result<Vec<Self>, ResourceParseError> {
        let mut res = vec![];
        for (name, values) in obj {
            let values = values
                .as_str()
                .ok_or_else(|| ResourceParseError::InvalidType(values.clone()))?
                .split('|')
                .map(Property::parse)
                .collect();
            res.push(MultipartBlockstate {
                name: name.to_owned(),
                values,
            });
        }
        Ok(res)
    }
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct ConfiguredModel {
    pub model: ResourceLocation<BlockModel>,
    #[serde(default)]
    pub x: f32,
    #[serde(default)]
    pub y: f32,
    #[serde(default)]
    pub uvlock: bool,
    #[serde(default = "default_weight")]
    pub weight: f32,
}

fn default_weight() -> f32 {
    1.0
}

impl ConfiguredModel {
    pub fn parse_model_pool(input: &Value) -> Result<Vec<ConfiguredModel>, ResourceParseError> {
        match input {
            Value::Array(arr) => Ok(arr
                .iter()
                .map(|variant| serde_json::from_value(variant.clone()))
                .collect::<Result<_, serde_json::Error>>()?),
            Value::Object(obj) => Ok(vec![serde_json::from_value(Value::Object(obj.clone()))?]),
            other => Err(ResourceParseError::InvalidType(other.clone())),
        }
    }
}
