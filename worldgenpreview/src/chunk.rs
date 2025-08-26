use std::{hash::BuildHasher, str::FromStr};

use bevy::{
    asset::RenderAssetUsages,
    log::tracing::instrument,
    math::{I16Vec3, U16Vec3, Vec2},
    platform::{collections::HashMap, hash::FixedHasher},
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
};
use bitflags::bitflags;
use libworldgen::{
    density_function::{
        FunctionOp, from_json,
        std_ops::{Add, Noise, YClampedGradient},
    },
    random::XoroshiroRng,
};
use mcpackloader::{
    ResourceLocation, ResourceParseError,
    blockstates::{
        Blockstate, BlockstateFile, ConfiguredModel, MultipartBlockstate, MultipartPredicate,
        Property, VariantBlockstate,
    },
    models::{self, BlockModel, Direction, Element, ElementRotation, RotationAxis},
    textures::Texture,
    worldgen::{DensityFunction, NoiseParameters},
};

use crate::{
    registry::{LoadedResource, Registries, RegistriesHandle, Registry, RegistryResource},
    textures::{ATLAS_SIZE, LoadedTexture},
};

const CHUNK_HEIGHT: usize = 348;
const BLOCKS_PER_CHUNK: usize = 16 * 16 * CHUNK_HEIGHT;

type BlockId = u8;

#[derive(Component)]
#[require(Mesh3d, MeshMaterial3d<StandardMaterial>)]
pub struct Chunk {
    palette: [(Block, LoadedBlock); u8::MAX as usize],

    /// Format: `[[[BlockId; CHUNK_X_SIZE]; CHUNK_Z_SIZE]; CHUNK_HEIGHT]`
    ///
    /// Indexing: `data[x + (16 * z) + (16 * 16 * y)]`
    data: [BlockId; BLOCKS_PER_CHUNK],
}

pub static FULL_BLOCK: &(Block, LoadedBlock) = &(
    Block::new(ResourceLocation::air(), Blockstate(HashMap::new())),
    LoadedBlock {
        mesh: CullableMeshSet::new(),
        full_block: true,
        never_culled: false,
        empty: false,
    },
);

impl Chunk {
    pub const fn air() -> (Block, LoadedBlock) {
        (
            Block::new(ResourceLocation::air(), Blockstate(HashMap::new())),
            LoadedBlock::new(),
        )
    }

    #[instrument(skip_all)]
    pub fn generate_test_chunk(registries: RegistriesHandle, pos: IVec2) -> Self {
        let mut palette = [const { Self::air() }; u8::MAX as usize];
        let mut registries = registries.lock();
        let stone = Block::new_with_state(ResourceLocation::new_mc("stone"), vec![]);
        palette[1] = stone
            .into_palette_format(registries.as_mut())
            .unwrap_or(Self::air());
        let grass_block = Block::new_with_state(
            ResourceLocation::new_mc("grass_block"),
            vec![("snowy".into(), Property::Bool(false))],
        );
        palette[2] = grass_block
            .into_palette_format(registries.as_mut())
            .unwrap_or(Self::air());

        let df = registries
            .get_or_load::<DensityFunction>(&ResourceLocation::new("worldgenpreview", "test"))
            .unwrap();
        let df = from_json(&df.0, &mut registries).unwrap();
        drop(registries);
        Self {
            palette,
            data: Self::generate_test_chunk_data(pos, df),
        }
    }

    #[instrument(skip_all)]
    pub fn generate_test_chunk_data(
        pos: IVec2,
        df: Box<dyn FunctionOp>,
    ) -> [BlockId; BLOCKS_PER_CHUNK] {
        let pos = pos * 16;

        let mut bin_layers = Vec::with_capacity(16 * 16 * 16 * (CHUNK_HEIGHT / 16) / 64);
        for y_chunk in 0..(CHUNK_HEIGHT / 16) {
            let subchunk_floats = df.run_subchunk(&glam::I64Vec3::new(
                pos.x as i64,
                (y_chunk as i64 * 16) - 64,
                pos.y as i64,
            ));
            let subchunk = subchunk_floats.chunks_exact(16 * 16).map(|data| {
                let mut res = [0u64; 4];
                for (int_i, res_int) in res.iter_mut().enumerate() {
                    let base_i = int_i * 64;
                    let mut int = 0;
                    for i in 0..64 {
                        int |= ((data[base_i + i] > 0.0) as u64) << i;
                    }
                    *res_int = int;
                }
                res
            });
            bin_layers.extend(subchunk);
        }
        let mut data = [0u8; BLOCKS_PER_CHUNK];
        let mut covered = [0u64; 4];
        for (y, layer) in bin_layers.iter().enumerate().rev() {
            let y_data_idx = y * 16 * 16;
            for i in 0..4 {
                let i_data_idx = i * 64;
                for bit in 0..64 {
                    if layer[i] >> bit & 1 == 1 {
                        let data_idx = y_data_idx + i_data_idx + bit;
                        if covered[i] >> bit & 1 == 1 {
                            data[data_idx] = 1u8;
                        } else {
                            data[data_idx] = 2u8;
                        }
                    }
                }
                covered[i] |= layer[i];
                covered[i] &= layer[i];
            }
        }
        data
    }

    #[instrument(skip_all)]
    pub fn example_chunk(registries: &mut Registries) -> Self {
        let mut palette = [const { Self::air() }; u8::MAX as usize];
        let stone = Block::new_with_state(ResourceLocation::new_mc("stone"), vec![]);
        palette[1] = stone.into_palette_format(registries).unwrap_or(Self::air());
        let dirt = Block::new_with_state(ResourceLocation::new_mc("dirt"), vec![]);
        palette[2] = dirt.into_palette_format(registries).unwrap_or(Self::air());
        let grass_block = Block::new_with_state(
            ResourceLocation::new_mc("grass_block"),
            vec![("snowy".into(), Property::Bool(false))],
        );
        palette[3] = grass_block
            .into_palette_format(registries)
            .unwrap_or(Self::air());

        let mut data = [0; BLOCKS_PER_CHUNK];
        for y in 0..121 {
            data[(16 * 16 * y)..(16 * 16 * (y + 1))].copy_from_slice(&[1; 16 * 16]);
        }
        for y in 121..125 {
            data[(16 * 16 * y)..(16 * 16 * (y + 1))].copy_from_slice(&[2; 16 * 16]);
        }
        data[(16 * 16 * 125)..(16 * 16 * (125 + 1))].copy_from_slice(&[3; 16 * 16]);

        let oak_log = Block::new_with_state(
            ResourceLocation::new_mc("oak_log"),
            vec![("axis".into(), Property::Enum("y".into()))],
        );
        palette[4] = oak_log
            .into_palette_format(registries)
            .unwrap_or(Self::air());
        let oak_leaves = Block::new_with_state(ResourceLocation::new_mc("oak_leaves"), vec![]);
        palette[5] = oak_leaves
            .into_palette_format(registries)
            .unwrap_or(Self::air());
        for y in 128..130 {
            for x in 6..11 {
                for z in 6..11 {
                    data[(16 * 16 * y) + (16 * z) + x] = 5;
                }
            }
        }
        for y in 130..132 {
            for x in 6..11 {
                for z in 6..11 {
                    if ((x as isize - 8).pow(2) + (z as isize - 8).pow(2)) < 4 {
                        data[(16 * 16 * y) + (16 * z) + x] = 5;
                    }
                }
            }
        }
        for y in 126..130 {
            data[(16 * 16 * y) + (16 * 8) + 8] = 4;
        }

        Chunk { palette, data }
    }

    pub fn index_unchecked(&self, at: &U16Vec3) -> &(Block, LoadedBlock) {
        let palette_id =
            self.data[at.x as usize + (16 * at.z as usize) + (const { 16 * 16 } * at.y as usize)];
        &self.palette[palette_id as usize]
    }

    pub fn index_or_full_block(&self, at: &I16Vec3) -> &(Block, LoadedBlock) {
        if !(0..16).contains(&at.x)
            || !(0..CHUNK_HEIGHT as i16).contains(&at.y)
            || !(0..16).contains(&at.z)
        {
            return FULL_BLOCK;
        }
        let palette_id =
            self.data[at.x as usize + (16 * at.z as usize) + (const { 16 * 16 } * at.y as usize)];
        &self.palette[palette_id as usize]
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DirectionBits: u8 {
        const UP = 1;
        const DOWN = 1 << 1;
        const NORTH = 1 << 2;
        const SOUTH = 1 << 3;
        const EAST = 1 << 4;
        const WEST = 1 << 5;
    }
}

impl DirectionBits {
    pub const fn from_each(
        up: bool,
        down: bool,
        north: bool,
        south: bool,
        east: bool,
        west: bool,
    ) -> Self {
        Self::from_bits_retain(
            up as u8
                | ((down as u8) << 1)
                | ((north as u8) << 2)
                | ((south as u8) << 3)
                | ((east as u8) << 4)
                | ((west as u8) << 5),
        )
    }
}

pub struct CulledChunk {
    chunk: Box<Chunk>,
    non_culled: Vec<(U16Vec3, DirectionBits)>,
}

impl CulledChunk {
    pub fn new(chunk: Box<Chunk>) -> Self {
        let non_culled = Self::gen_chunk_culling_data(&chunk);
        Self { chunk, non_culled }
    }

    #[instrument(skip_all)]
    pub fn to_mesh(&self) -> Result<CullableMesh, ResourceParseError> {
        let mut res = CullableMesh::new();
        for (pos, faces) in &self.non_culled {
            let block = &self.chunk.index_unchecked(pos).1;
            block.to_mesh(&mut res, *faces, &pos.as_vec3());
        }
        Ok(res)
    }

    #[instrument(skip_all)]
    fn gen_chunk_culling_data(chunk: &Chunk) -> Vec<(U16Vec3, DirectionBits)> {
        let mut non_culled = vec![];

        for y in 1..(CHUNK_HEIGHT as u16 - 1) {
            for x in 1..15 {
                for z in 1..15 {
                    Self::try_inner_cull(chunk, U16Vec3::new(x, y, z), &mut non_culled);
                }
                Self::try_edge_cull(chunk, U16Vec3::new(x, y, 0), &mut non_culled);
                Self::try_edge_cull(chunk, U16Vec3::new(x, y, 15), &mut non_culled);
            }
            for z in 0..16 {
                Self::try_edge_cull(chunk, U16Vec3::new(0, y, z), &mut non_culled);
                Self::try_edge_cull(chunk, U16Vec3::new(15, y, z), &mut non_culled);
            }
        }
        for x in 0..16 {
            for z in 0..16 {
                Self::try_edge_cull(chunk, U16Vec3::new(x, 0, z), &mut non_culled);
                Self::try_edge_cull(
                    chunk,
                    U16Vec3::new(x, CHUNK_HEIGHT as u16 - 1, z),
                    &mut non_culled,
                );
            }
        }

        // info!("not culled: {}", non_culled.len());

        non_culled
    }

    fn try_inner_cull(chunk: &Chunk, pos: U16Vec3, non_culled: &mut Vec<(U16Vec3, DirectionBits)>) {
        let block = chunk.index_unchecked(&pos);
        let non_empty = !block.1.empty;
        if non_empty {
            let dirs = Self::gen_block_culling_data_unchecked(chunk, &pos);
            if (!dirs.is_empty()) || block.1.never_culled {
                non_culled.push((pos, dirs));
            }
        }
    }

    fn try_edge_cull(chunk: &Chunk, pos: U16Vec3, non_culled: &mut Vec<(U16Vec3, DirectionBits)>) {
        let non_empty = !chunk.index_unchecked(&pos).1.empty;
        if non_empty {
            let dirs = Self::gen_block_culling_data(chunk, &pos.as_i16vec3());
            non_culled.push((pos, dirs));
        }
    }

    fn gen_block_culling_data_unchecked(chunk: &Chunk, block: &U16Vec3) -> DirectionBits {
        DirectionBits::from_each(
            !chunk
                .index_unchecked(&U16Vec3::new(block.x, block.y + 1, block.z))
                .1
                .full_block,
            !chunk
                .index_unchecked(&U16Vec3::new(block.x, block.y - 1, block.z))
                .1
                .full_block,
            !chunk
                .index_unchecked(&U16Vec3::new(block.x, block.y, block.z - 1))
                .1
                .full_block,
            !chunk
                .index_unchecked(&U16Vec3::new(block.x, block.y, block.z + 1))
                .1
                .full_block,
            !chunk
                .index_unchecked(&U16Vec3::new(block.x + 1, block.y, block.z))
                .1
                .full_block,
            !chunk
                .index_unchecked(&U16Vec3::new(block.x - 1, block.y, block.z))
                .1
                .full_block,
        )
    }

    fn gen_block_culling_data(chunk: &Chunk, block: &I16Vec3) -> DirectionBits {
        DirectionBits::from_each(
            !chunk
                .index_or_full_block(&I16Vec3::new(block.x, block.y + 1, block.z))
                .1
                .full_block,
            !chunk
                .index_or_full_block(&I16Vec3::new(block.x, block.y - 1, block.z))
                .1
                .full_block,
            !chunk
                .index_or_full_block(&I16Vec3::new(block.x, block.y, block.z - 1))
                .1
                .full_block,
            !chunk
                .index_or_full_block(&I16Vec3::new(block.x, block.y, block.z + 1))
                .1
                .full_block,
            !chunk
                .index_or_full_block(&I16Vec3::new(block.x + 1, block.y, block.z))
                .1
                .full_block,
            !chunk
                .index_or_full_block(&I16Vec3::new(block.x - 1, block.y, block.z))
                .1
                .full_block,
        )
    }
}

pub struct Block {
    name: ResourceLocation<BlockstateFile>,
    blockstate: Blockstate,
}

impl Block {
    pub const fn new(name: ResourceLocation<BlockstateFile>, blockstate: Blockstate) -> Self {
        Self { name, blockstate }
    }

    pub fn new_with_state(
        name: ResourceLocation<BlockstateFile>,
        blockstate: Vec<(String, Property)>,
    ) -> Self {
        Self {
            name,
            blockstate: Blockstate(HashMap::from_iter(blockstate)),
        }
    }

    pub fn into_palette_format(
        self,
        registries: &mut Registries,
    ) -> Result<(Self, LoadedBlock), ResourceParseError> {
        let loaded_block = self.load_configured_block(registries)?;
        Ok((self, loaded_block))
    }

    #[instrument(skip_all)]
    pub fn load_configured_block(
        &self,
        registries: &mut Registries,
    ) -> Result<LoadedBlock, ResourceParseError> {
        let blockstate_file = registries
            .get_or_load::<BlockstateFile>(&self.name)?
            .clone();

        match blockstate_file {
            BlockstateFile::Variants(variants) => {
                for (predicates, variant_pool) in variants {
                    if !matches_variant(self, &predicates)? {
                        continue;
                    }

                    let configured_model = variant_pool.first().ok_or_else(|| {
                        format!(
                            "Empty variant pool in blockstate file `{}`, predicates: {:?}",
                            self.name, predicates
                        )
                    })?;

                    return LoadedBlock::from_configured_model(
                        configured_model,
                        registries,
                        &self.name,
                        &self.blockstate,
                    );
                }
                Err(format!(
                    "No variant matches in blockstate file for block `{}`, blockstate: {:?}",
                    self.name, self.blockstate
                )
                .into())
            }
            BlockstateFile::Multipart(multipart) => {
                let mut configured_block = LoadedBlock::new();
                for (predicates, variant_pool) in multipart {
                    if matches_multipart(self, &predicates)? {
                        let configured_model = variant_pool.first().ok_or_else(|| {
                            format!(
                                "Empty variant pool in blockstate file `{}`, predicates: {:?}",
                                self.name, predicates
                            )
                        })?;

                        configured_block.add(LoadedBlock::from_configured_model(
                            configured_model,
                            registries,
                            &self.name,
                            &self.blockstate,
                        )?);
                    }
                }
                Ok(configured_block)
            }
        }
    }
}

#[derive(Default, Clone)]
pub struct LoadedBlock {
    pub mesh: CullableMeshSet,
    pub full_block: bool,
    pub never_culled: bool,
    pub empty: bool,
}

impl LoadedBlock {
    /// Creates a new empty block
    pub const fn new() -> Self {
        Self {
            mesh: CullableMeshSet::new(),
            full_block: false,
            never_culled: false,
            empty: true,
        }
    }

    #[instrument(skip_all)]
    pub fn from_configured_model(
        configured_model: &ConfiguredModel,
        registries: &mut Registries,
        block_name: &ResourceLocation<BlockstateFile>,
        state: &Blockstate,
    ) -> Result<Self, ResourceParseError> {
        let mut model = registries
            .get_or_load::<LoadedBlockModel>(&configured_model.model)?
            .clone();

        let mesh = model.mesh.as_mut().ok_or_else(|| {
            format!(
                "Block Model `{block_name}` with state `{state:?}` has no mesh, nor do any of it's parents have a mesh"
            )
        })?;

        if configured_model.x != 0.0 {
            mesh.rotate(&ElementRotation {
                origin: models::Vec3(8.0, 8.0, 8.0),
                axis: RotationAxis::X,
                angle: configured_model.x,
                rescale: false,
            });
        }
        if configured_model.y != 0.0 {
            mesh.rotate(&ElementRotation {
                origin: models::Vec3(8.0, 8.0, 8.0),
                axis: RotationAxis::Y,
                angle: configured_model.y,
                rescale: false,
            });
        }

        let never_culled = !mesh.never_culled.faces.is_empty();

        let resolved_textures = model
            .textures
            .iter()
            .map(|(id, tx)| {
                Ok((
                    *id,
                    registries
                        .get(&resolve_texture(&model.textures, tx).unwrap())
                        .unwrap()?,
                ))
            })
            .collect::<Result<HashMap<_, _>, ResourceParseError>>()?;

        mesh.map_uvs(&resolved_textures);

        Ok(LoadedBlock {
            mesh: model.mesh.unwrap(),
            full_block: model.full_block.unwrap(),
            never_culled,
            empty: false,
        })
    }

    pub fn add(&mut self, other: LoadedBlock) {
        // self.parts.extend_from_slice(&other.parts);
        self.mesh.extend(&other.mesh);
        self.full_block |= other.full_block;
        self.never_culled |= other.never_culled;
        self.empty &= other.empty;
    }

    pub fn to_mesh(&self, to: &mut CullableMesh, faces: DirectionBits, pos: &Vec3) {
        self.mesh.merge_to(to, faces, pos);
    }
}

fn matches_variant(
    block: &Block,
    predicates: &[VariantBlockstate],
) -> Result<bool, ResourceParseError> {
    for predicate in predicates {
        let prop = block.blockstate.0.get(&predicate.name).ok_or_else(|| {
            format!(
                "Expected block `{}` to have a blockstate property `{}`",
                block.name, predicate.name
            )
        })?;
        if &predicate.value != prop {
            return Ok(false);
        }
    }

    Ok(true)
}

fn matches_multipart(
    block: &Block,
    predicate: &MultipartPredicate,
) -> Result<bool, ResourceParseError> {
    match predicate {
        MultipartPredicate::Always => Ok(true),
        MultipartPredicate::And(predicates) => {
            for predicate in predicates {
                if !matches_multipart_blockstate(block, predicate)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        MultipartPredicate::Or(predicates) => {
            for predicate in predicates {
                if matches_multipart_blockstate(block, predicate)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        MultipartPredicate::Single(predicate) => matches_multipart_blockstate(block, predicate),
    }
}

fn matches_multipart_blockstate(
    block: &Block,
    predicates: &[MultipartBlockstate],
) -> Result<bool, ResourceParseError> {
    for predicate in predicates {
        let prop = block.blockstate.0.get(&predicate.name).ok_or_else(|| {
            format!(
                "Expected block `{}` to have a blockstate property `{}`",
                block.name, predicate.name
            )
        })?;

        if !predicate.values.iter().any(|value| prop == value) {
            return Ok(false);
        }
    }

    Ok(true)
}

#[derive(Clone)]
pub struct LoadedBlockModel {
    pub mesh: Option<CullableMeshSet>,
    pub textures: HashMap<u64, String>,
    pub full_block: Option<bool>,
}

impl LoadedResource for LoadedBlockModel {
    type Resource = BlockModel;

    fn load(
        res_loc: &ResourceLocation<BlockModel>,
        res: BlockModel,
        registries: &mut Registries,
    ) -> std::result::Result<Self, ResourceParseError> {
        let mut mesh = None;
        let mut textures = HashMap::new();
        let mut full_block = None;
        if let Some(parent) = &res.parent {
            let parent_model: &LoadedBlockModel = registries.get_or_load(parent)?;
            if let Some(parent_mesh) = &parent_model.mesh {
                mesh = Some(parent_mesh.clone());
                full_block = Some(parent_model.full_block.unwrap());
            }
            textures.extend(parent_model.textures.iter().map(|(k, v)| (*k, v.clone())));
        }

        if let Some(elements) = res.elements {
            let mut new_mesh = CullableMeshSet::new();
            for element in elements {
                new_mesh.push_new_cuboid(element);
            }
            mesh = Some(new_mesh);

            full_block = Some(res_loc == &ResourceLocation::new_mc("block/cube"));
        }

        if let Some(model_textures) = res.textures {
            textures.extend(
                model_textures
                    .iter()
                    .map(|(k, v)| (create_texture_id(&format!("#{k}")), v.clone())),
            );

            // preload textures
            for texture in model_textures.values() {
                if !texture.starts_with('#') {
                    registries.get_or_load::<LoadedTexture>(
                        &ResourceLocation::from_str(texture).unwrap(),
                    )?;
                }
            }
        }

        Ok(LoadedBlockModel {
            mesh,
            textures,
            full_block,
        })
    }
}

impl RegistryResource for LoadedBlockModel {
    fn get(registries: &Registries) -> &Registry<Self> {
        &registries.models
    }

    fn get_mut(registries: &mut Registries) -> &mut Registry<Self> {
        &mut registries.models
    }
}

/// A mesh split into parts to simplify culling.
///
/// Only meshes will be displayed where there is no block in the given direction, except for `never_culled`, which is always displayed.
#[derive(Default, Clone)]
pub struct CullableMeshSet {
    up: CullableMesh,
    down: CullableMesh,
    north: CullableMesh,
    south: CullableMesh,
    west: CullableMesh,
    east: CullableMesh,
    never_culled: CullableMesh,
}

fn resolve_texture(
    textures: &HashMap<u64, String>,
    texture: &str,
) -> Option<ResourceLocation<Texture>> {
    Some(
        ResourceLocation::from_str(if texture.starts_with('#') {
            let mut texture_str = textures.get(&create_texture_id(texture))?;
            while texture_str.starts_with('#') {
                dbg!(texture_str);
                texture_str = textures.get(&create_texture_id(texture_str))?;
            }
            texture_str
        } else {
            texture
        })
        .unwrap(),
    )
}

impl CullableMeshSet {
    pub const fn new() -> Self {
        Self {
            up: CullableMesh::new(),
            down: CullableMesh::new(),
            north: CullableMesh::new(),
            south: CullableMesh::new(),
            west: CullableMesh::new(),
            east: CullableMesh::new(),
            never_culled: CullableMesh::new(),
        }
    }

    pub fn merge_to(&self, merged_mesh: &mut CullableMesh, faces: DirectionBits, pos: &Vec3) {
        let prev_len = merged_mesh.faces.len();
        if faces.contains(DirectionBits::UP) {
            merged_mesh.extend(&self.up);
        }
        if faces.contains(DirectionBits::DOWN) {
            merged_mesh.extend(&self.down);
        }

        if faces.contains(DirectionBits::NORTH) {
            merged_mesh.extend(&self.north);
        }

        if faces.contains(DirectionBits::SOUTH) {
            merged_mesh.extend(&self.south);
        }

        if faces.contains(DirectionBits::EAST) {
            merged_mesh.extend(&self.east);
        }

        if faces.contains(DirectionBits::WEST) {
            merged_mesh.extend(&self.west);
        }
        for face in &mut merged_mesh.faces[prev_len..] {
            face.translate(pos);
        }
    }

    pub fn push_new_cuboid(&mut self, cuboid: Element) {
        let from = cuboid.from.to_glam() / 16.0;
        let to = cuboid.to.to_glam() / 16.0;
        for (dir, face) in &cuboid.faces {
            let from_proj = match dir {
                Direction::Up | Direction::Down => Vec2::new(from.x, from.z),
                Direction::North | Direction::South => Vec2::new(from.x, from.y),
                Direction::East | Direction::West => Vec2::new(from.z, from.y),
            };
            let to_proj = match dir {
                Direction::Up | Direction::Down => Vec2::new(to.x, to.z),
                Direction::North | Direction::South => Vec2::new(to.x, to.y),
                Direction::East | Direction::West => Vec2::new(to.z, to.y),
            };

            // Compute size of the face
            let size = to_proj - from_proj;

            // UV rectangle
            let uv = if let Some([x1, y1, x2, y2]) = face.uv {
                Rect {
                    min: Vec2::new(x1 as f32, y1 as f32),
                    max: Vec2::new(x2 as f32, y2 as f32),
                }
            } else {
                Rect {
                    min: from_proj * 16.0,
                    max: to_proj * 16.0,
                }
            };

            // UV rotation
            let uv_rotation = face.rotation as u32;

            // Tint color
            let color = match face.tintindex {
                -1 => Vertex::TINT_INDEX_N1,
                0 => Vertex::TINT_INDEX_0,
                1 => Vertex::TINT_INDEX_1,
                _ => Vertex::TINT_INDEX_N1,
            };

            // Create rect
            let mut new_face = Face::new(
                size,
                dir.clone(),
                uv,
                uv_rotation,
                color,
                create_texture_id(&face.texture),
            );

            // Translate to correct position
            let offset = match dir {
                Direction::Up => Vec3::new(from.x, to.y, from.z),
                Direction::Down => Vec3::new(from.x, from.y, from.z),
                Direction::North => Vec3::new(from.x, from.y, from.z),
                Direction::South => Vec3::new(from.x, from.y, to.z),
                Direction::East => Vec3::new(to.x, from.y, from.z),
                Direction::West => Vec3::new(from.x, from.y, from.z),
            };
            new_face.translate(&offset);

            // Apply rotation if any
            if let Some(rotation) = &cuboid.rotation {
                new_face.rotate(rotation);
            }

            // Decide which mesh to push into
            if face.cullface.is_none() {
                self.never_culled.push_face(new_face);
            } else {
                let mesh = match face.cullface.as_ref().unwrap() {
                    Direction::Up => &mut self.up,
                    Direction::Down => &mut self.down,
                    Direction::North => &mut self.north,
                    Direction::South => &mut self.south,
                    Direction::East => &mut self.east,
                    Direction::West => &mut self.west,
                };
                mesh.push_face(new_face);
            }
        }
    }

    pub fn rotate(&mut self, by: &ElementRotation) {
        self.up.rotate(by);
        self.down.rotate(by);
        self.north.rotate(by);
        self.south.rotate(by);
        self.east.rotate(by);
        self.west.rotate(by);
        self.never_culled.rotate(by);
    }
    pub fn map_uvs(&mut self, textures: &HashMap<u64, &LoadedTexture>) {
        self.up.map_uvs(textures);
        self.down.map_uvs(textures);
        self.north.map_uvs(textures);
        self.south.map_uvs(textures);
        self.east.map_uvs(textures);
        self.west.map_uvs(textures);
        self.never_culled.map_uvs(textures);
    }

    pub fn extend(&mut self, by: &CullableMeshSet) {
        self.up.extend(&by.up);
        self.down.extend(&by.down);
        self.north.extend(&by.north);
        self.south.extend(&by.south);
        self.east.extend(&by.east);
        self.west.extend(&by.west);
        self.never_culled.extend(&by.never_culled);
    }
}

#[derive(Default, Clone)]
pub struct CullableMesh {
    faces: Vec<Face>,
}

fn map_uv(uv1: [f32; 2], uv2: &Vec2) -> [f32; 2] {
    [
        (uv1[0] + uv2.x) / ATLAS_SIZE as f32,
        (uv1[1] + uv2.y) / ATLAS_SIZE as f32,
    ]
}

impl CullableMesh {
    pub const fn new() -> Self {
        Self { faces: vec![] }
    }

    pub fn push_face(&mut self, face: Face) {
        self.faces.push(face);
    }

    pub fn rotate(&mut self, by: &ElementRotation) {
        for rect in &mut self.faces {
            rect.rotate(by);
        }
    }

    pub fn map_uvs(&mut self, textures: &HashMap<u64, &LoadedTexture>) {
        for face in &mut self.faces {
            face.map_uvs(textures);
        }
    }

    pub fn generate_indices(&self) -> Vec<u16> {
        let mut i = 0;
        let mut indices = Vec::with_capacity(self.faces.len() * 6);
        for _ in 0..self.faces.len() {
            indices.extend_from_slice(&[i, i + 1, i + 3, i + 2, i + 3, i + 1]);
            i += 4;
        }
        indices
    }

    pub fn extend(&mut self, other: &CullableMesh) {
        self.faces.extend_from_slice(&other.faces);
    }

    pub fn get_positions(&self) -> Vec<[f32; 3]> {
        let mut res = vec![[0.0, 0.0, 0.0]; self.faces.len() * 4];
        let mut i = 0;
        for face in &self.faces {
            res[i] = face.mesh[0].position;
            res[i + 1] = face.mesh[1].position;
            res[i + 2] = face.mesh[2].position;
            res[i + 3] = face.mesh[3].position;
            i += 4;
        }
        res
    }

    pub fn get_normals(&self) -> Vec<[f32; 3]> {
        let mut res = vec![[0.0, 0.0, 0.0]; self.faces.len() * 4];
        let mut i = 0;
        for face in &self.faces {
            let normal = normal_vector(&face.normal);
            res[i] = normal;
            res[i + 1] = normal;
            res[i + 2] = normal;
            res[i + 3] = normal;
            i += 4;
        }
        res
    }

    pub fn get_premapped_uvs(&self) -> Vec<[f32; 2]> {
        let mut res = vec![[0.0, 0.0]; self.faces.len() * 4];
        let mut i = 0;
        for face in &self.faces {
            res[i] = face.mesh[0].uv;
            res[i + 1] = face.mesh[1].uv;
            res[i + 2] = face.mesh[2].uv;
            res[i + 3] = face.mesh[3].uv;
            i += 4;
        }
        res
    }

    pub fn get_colors(&self) -> Vec<[f32; 4]> {
        let mut res = vec![[0.0, 0.0, 0.0, 0.0]; self.faces.len() * 4];
        let mut i = 0;
        for face in &self.faces {
            let color = face.color;
            res[i] = color;
            res[i + 1] = color;
            res[i + 2] = color;
            res[i + 3] = color;
            i += 4;
        }
        res
    }

    /// Creates a new bevy mesh, expects this mesh to already be uv-mapped using [`Self::map_uvs`]
    #[instrument(skip_all)]
    pub fn to_bevy_mesh(&self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.get_positions())
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.get_normals())
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.get_premapped_uvs())
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.get_colors())
        .with_inserted_indices(Indices::U16(self.generate_indices()))
    }
}

#[derive(Debug, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    const TINT_INDEX_N1: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    const TINT_INDEX_0: [f32; 4] = [0.4, 0.9, 0.2, 1.0];
    const TINT_INDEX_1: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

    pub fn translate(&mut self, by: &Vec3) {
        self.position[0] += by.x;
        self.position[1] += by.y;
        self.position[2] += by.z;
    }

    pub fn rotate(&mut self, by: &ElementRotation) {
        let angle_rad = by.angle.to_radians();
        let origin = by.origin.to_glam();

        let (sin, cos) = angle_rad.sin_cos();

        let pos = Vec3::new(self.position[0], self.position[1], self.position[2]) - origin;

        let rotated = match by.axis {
            RotationAxis::X => {
                Vec3::new(pos.x, pos.y * cos - pos.z * sin, pos.y * sin + pos.z * cos)
            }
            RotationAxis::Y => {
                Vec3::new(pos.x * cos + pos.z * sin, pos.y, -pos.x * sin + pos.z * cos)
            }
            RotationAxis::Z => {
                Vec3::new(pos.x * cos - pos.y * sin, pos.x * sin + pos.y * cos, pos.z)
            }
        };

        let final_pos = rotated + origin;
        self.position = [final_pos.x, final_pos.y, final_pos.z];
    }
}

#[derive(Debug, Clone)]
pub struct Face {
    /// The vertices in a rect must come in the following order:
    ///
    /// - top-left
    /// - bottom-left
    /// - bottom-right
    /// - top-right
    mesh: [Vertex; 4],
    normal: Direction,
    color: [f32; 4],
    texture_id: u64,
}

impl Face {
    pub fn new(
        size: Vec2,
        normal: Direction,
        uv: Rect,
        uv_rotation: u32,
        color: [f32; 4],
        texture_id: u64,
    ) -> Self {
        // Helper closure for vertex creation
        let v = |position: [f32; 3], uv: [f32; 2]| Vertex { position, uv };

        let uv_corners = [
            [uv.min.x, uv.max.y],
            [uv.min.x, uv.min.y],
            [uv.max.x, uv.min.y],
            [uv.max.x, uv.max.y],
        ];

        // Permute UVs based on rotation (0, 90, 180, 270)
        let perm = match uv_rotation % 360 {
            0 => [0, 1, 2, 3],
            90 => [2, 0, 3, 1],
            180 => [3, 2, 1, 0],
            270 => [1, 3, 0, 2],
            _ => [0, 1, 2, 3], // fallback to 0
        };
        let uv_rotated = [
            uv_corners[perm[0]],
            uv_corners[perm[1]],
            uv_corners[perm[2]],
            uv_corners[perm[3]],
        ];

        //                                         ignore the comments here \/ \/ lol
        let mesh = match normal {
            Direction::Up => [
                v([size.x, 0.0, size.y], uv_rotated[0]), // top-left (+Z)
                v([size.x, 0.0, 0.0], uv_rotated[1]),    // bottom-left (-Z)
                v([0.0, 0.0, 0.0], uv_rotated[2]),       // bottom-right (-Z)
                v([0.0, 0.0, size.y], uv_rotated[3]),    // top-right (+Z)
            ],
            Direction::Down => [
                v([0.0, 0.0, size.y], uv_rotated[0]),    // top-left (-Z)
                v([0.0, 0.0, 0.0], uv_rotated[1]),       // bottom-left (+Z)
                v([size.x, 0.0, 0.0], uv_rotated[2]),    // bottom-right (+Z)
                v([size.x, 0.0, size.y], uv_rotated[3]), // top-right (-Z)
            ],
            Direction::North => [
                v([0.0, 0.0, 0.0], uv_rotated[0]),       // bottom-left (+Y)
                v([0.0, size.y, 0.0], uv_rotated[1]),    // bottom-right (-Y)
                v([size.x, size.y, 0.0], uv_rotated[2]), // top-right (-Y)
                v([size.x, 0.0, 0.0], uv_rotated[3]),    // top-left (+Y)
            ],
            Direction::South => [
                v([size.x, 0.0, 0.0], uv_rotated[0]),    // bottom-left (+Y)
                v([size.x, size.y, 0.0], uv_rotated[1]), // bottom-right (-Y)
                v([0.0, size.y, 0.0], uv_rotated[2]),    // top-right (-Y)
                v([0.0, 0.0, 0.0], uv_rotated[3]),       // top-left (+Y)
            ],
            Direction::East => [
                v([0.0, 0.0, 0.0], uv_rotated[0]),       // bottom-left (+Y)
                v([0.0, size.y, 0.0], uv_rotated[1]),    // bottom-right (-Y)
                v([0.0, size.y, size.x], uv_rotated[2]), // top-right (-Y)
                v([0.0, 0.0, size.x], uv_rotated[3]),    // top-left (+Y)
            ],
            Direction::West => [
                v([0.0, 0.0, size.x], uv_rotated[0]),    // bottom-left (+Y)
                v([0.0, size.y, size.x], uv_rotated[1]), // bottom-right (-Y)
                v([0.0, size.y, 0.0], uv_rotated[2]),    // top-right (-Y)
                v([0.0, 0.0, 0.0], uv_rotated[3]),       // top-left (+Y)
            ],
        };

        Face {
            mesh,
            normal,
            color,
            texture_id,
        }
    }

    pub fn rotate(&mut self, by: &ElementRotation) {
        self.mesh[0].rotate(by);
        self.mesh[1].rotate(by);
        self.mesh[2].rotate(by);
        self.mesh[3].rotate(by);
    }

    pub fn translate(&mut self, by: &Vec3) {
        self.mesh[0].translate(by);
        self.mesh[1].translate(by);
        self.mesh[2].translate(by);
        self.mesh[3].translate(by);
    }

    pub fn map_uvs(&mut self, textures: &HashMap<u64, &LoadedTexture>) {
        let base = &textures[&self.texture_id].atlas_pos;
        self.mesh[0].uv = map_uv(self.mesh[0].uv, base);
        self.mesh[1].uv = map_uv(self.mesh[1].uv, base);
        self.mesh[2].uv = map_uv(self.mesh[2].uv, base);
        self.mesh[3].uv = map_uv(self.mesh[3].uv, base);
    }
}

fn normal_vector(direction: &Direction) -> [f32; 3] {
    match direction {
        Direction::Up => [0.0, 1.0, 0.0],
        Direction::Down => [0.0, -1.0, 0.0],
        Direction::North => [0.0, 0.0, -1.0],
        Direction::South => [0.0, 0.0, 1.0],
        Direction::East => [1.0, 0.0, 0.0],
        Direction::West => [-1.0, 0.0, 0.0],
    }
}

fn create_texture_id(texture: &str) -> u64 {
    FixedHasher.hash_one(texture)
}
