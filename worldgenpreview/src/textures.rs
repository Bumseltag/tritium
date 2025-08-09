use std::result::Result;

use bevy::{
    asset::RenderAssetUsages,
    image::{DynamicTextureAtlasBuilderError, ImageSampler},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use guillotiere::Size;
use mcpackloader::{ResourceLocation, ResourceParseError, textures::Texture};

use crate::registry::{FromLoader, LoadedResource, Registries, Registry};

pub struct LoadedTexture {
    pub atlas_pos: Vec2,
}

impl LoadedResource for LoadedTexture {
    type Resource = Texture;

    async fn load(
        _res_loc: &ResourceLocation<Texture>,
        res: Texture,
        registries: &mut Registries,
    ) -> Result<Self, ResourceParseError> {
        let size = Size::new(res.sprite.width() as i32, res.sprite.height() as i32);
        let Some(alloc) = registries.atlas_alloc.allocate(size) else {
            return Err("Not enough space in atlas".into());
        };
        registries
            .from_loader_send
            .send(FromLoader::LoadTextureIntoAtlas(Box::new(res.sprite)))
            .await
            .map_err(|err| err.to_string())?;
        let pos = alloc.rectangle.min;
        Ok(LoadedTexture {
            atlas_pos: Vec2::new(pos.x as f32, pos.y as f32),
        })
    }

    fn get(registries: &Registries) -> &Registry<Self> {
        &registries.textures
    }

    fn get_mut(registries: &mut Registries) -> &mut Registry<Self> {
        &mut registries.textures
    }
}

pub const ATLAS_SIZE: u32 = 512;

#[derive(Resource)]
pub struct DynamicTextureAtlas {
    pub atlas: Handle<Image>,
    pub layout: Handle<TextureAtlasLayout>,
    pub builder: DynamicTextureAtlasBuilder,
    pub material: Handle<StandardMaterial>,
}

impl DynamicTextureAtlas {
    pub fn add(
        &mut self,
        image: &Image,
        layouts: &mut Assets<TextureAtlasLayout>,
        images: &mut Assets<Image>,
    ) -> Result<usize, DynamicTextureAtlasBuilderError> {
        self.builder.add_texture(
            layouts.get_mut(&self.layout).unwrap(),
            image,
            images.get_mut(&self.atlas).unwrap(),
        )
    }
}

impl FromWorld for DynamicTextureAtlas {
    fn from_world(world: &mut World) -> Self {
        let mut images = world.resource_mut::<Assets<Image>>();
        let mut image = Image::new_fill(
            Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0, 0, 0, 0],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        image.sampler = ImageSampler::nearest();
        let atlas = images.add(image);
        let mut layouts = world.resource_mut::<Assets<TextureAtlasLayout>>();
        let layout = layouts.add(TextureAtlasLayout::new_empty(uvec2(ATLAS_SIZE, ATLAS_SIZE)));
        let mut materials = world.resource_mut::<Assets<StandardMaterial>>();
        let material = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(atlas.clone()),
            reflectance: 0.0,
            alpha_mode: AlphaMode::Mask(0.5),
            ..Default::default()
        });
        Self {
            atlas,
            layout,
            builder: DynamicTextureAtlasBuilder::new(uvec2(ATLAS_SIZE, ATLAS_SIZE), 0),
            material,
        }
    }
}
