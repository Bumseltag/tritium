use std::{fs::File, io::BufReader, path::PathBuf};

use hashbrown::HashMap;
use serde::Deserialize;

use crate::{ResourceLocation, ResourceParseError, ResourceType};

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct BlockModel {
    pub parent: Option<ResourceLocation<BlockModel>>,
    pub textures: Option<HashMap<String, String>>,
    pub elements: Option<Vec<Element>>,
}

impl ResourceType for BlockModel {
    fn to_path(res_loc: &ResourceLocation<Self>) -> PathBuf {
        res_loc.to_assets_path("models", "json")
    }

    fn open(path: PathBuf) -> Result<Self, ResourceParseError> {
        Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
    }
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct Element {
    pub from: Vec3,
    pub to: Vec3,
    pub rotation: Option<ElementRotation>,
    pub faces: HashMap<Direction, Face>,
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct ElementRotation {
    pub origin: Vec3,
    pub axis: RotationAxis,
    pub angle: f32,
    #[serde(default)]
    pub rescale: bool,
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum RotationAxis {
    #[serde(rename = "x")]
    X,
    #[serde(rename = "y")]
    Y,
    #[serde(rename = "z")]
    Z,
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct Vec3(pub f32, pub f32, pub f32);

impl Vec3 {
    #[cfg(feature = "glam")]
    pub fn to_glam(&self) -> glam::Vec3 {
        glam::Vec3::new(self.0, self.1, self.2)
    }
}

#[cfg(feature = "glam")]
impl From<Vec3> for glam::Vec3 {
    fn from(val: Vec3) -> Self {
        Vec3::to_glam(&val)
    }
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Face {
    pub uv: Option<[u8; 4]>,
    pub texture: String,
    pub cullface: Option<Direction>,
    #[serde(default)]
    pub rotation: u16,
    #[serde(default = "default_tintindex")]
    pub tintindex: i32,
}

fn default_tintindex() -> i32 {
    -1
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Direction {
    #[serde(rename = "down")]
    Down,
    #[serde(rename = "up")]
    Up,
    #[serde(rename = "north")]
    North,
    #[serde(rename = "south")]
    South,
    #[serde(rename = "west")]
    West,
    #[serde(rename = "east")]
    East,
}
