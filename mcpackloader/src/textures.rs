use std::{fs::File, io::BufReader, path::PathBuf};

use image::{ImageFormat, RgbaImage};

use crate::{ResourceLocation, ResourceParseError, ResourceType, Ticks};

pub struct Texture {
    pub sprite: RgbaImage,
    pub animation: Option<TextureAnimation>,
}

impl ResourceType for Texture {
    fn to_path(res_loc: &ResourceLocation<Self>) -> PathBuf {
        res_loc.to_assets_path("textures", "png")
    }

    fn open(path: PathBuf) -> Result<Self, ResourceParseError> {
        let file = File::open(path)?;
        let sprite = image::load(BufReader::new(file), ImageFormat::Png)?;

        Ok(Self {
            sprite: sprite.to_rgba8(),
            animation: None,
        })
    }
}

pub struct TextureAnimation {
    pub interpolate: bool,
    pub frametime: Ticks,
    pub frames: Option<Vec<u32>>,
}

impl Default for TextureAnimation {
    fn default() -> Self {
        Self {
            interpolate: false,
            frametime: Ticks::new(1),
            frames: None,
        }
    }
}
