use std::any::Any;
use std::ffi::OsStr;
use std::fs::File;
use std::hash::{BuildHasher, Hash, Hasher};
use std::io;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::result::Result;
use std::sync::{Arc, Mutex, MutexGuard};
use std::{any::type_name, env, fs, future::poll_fn, path::PathBuf, str::FromStr, task::Poll};

use bevy::platform::hash::FixedHasher;
use bevy::prelude::*;
use bevy::tasks::{Task, block_on};
use bevy::{
    asset::{Assets, RenderAssetUsages},
    ecs::{
        resource::Resource,
        system::{Commands, ResMut},
    },
    image::{Image, TextureAtlasLayout},
    math::IVec2,
    pbr::{MeshMaterial3d, StandardMaterial},
    platform::collections::{HashMap, HashSet},
    render::mesh::{Mesh, Mesh3d},
    tasks::AsyncComputeTaskPool,
    transform::components::Transform,
};
use crossbeam_channel::{self, Receiver, Sender};
use guillotiere::{AtlasAllocator, euclid::Size2D};
use image::{DynamicImage, RgbaImage};
use libworldgen::density_function::std_ops::register_std_ops;
use libworldgen::registry::ResourceLoader;
use mcpackloader::worldgen::{DensityFunction, NoiseParameters};
use mcpackloader::{
    ResourceLocation, ResourceParseError, ResourceType, blockstates::BlockstateFile,
};
use tracing::{error, info_span, instrument};
use zip::ZipArchive;

use crate::{
    AppState,
    chunk::{Chunk, CulledChunk, LoadedBlockModel},
    textures::{ATLAS_SIZE, DynamicTextureAtlas, LoadedTexture},
};

#[derive(Resource, Clone)]
pub struct RegistriesHandle(Arc<Mutex<libworldgen::registry::Registries>>);

impl RegistriesHandle {
    fn new(packs: Vec<PathBuf>, from_loader_send: Sender<FromLoader>) -> Self {
        let mut reg = libworldgen::registry::Registries::new(Box::new(Registries::new(
            packs,
            from_loader_send,
        )));

        register_std_ops(&mut reg);

        Self(Arc::new(Mutex::new(reg)))
    }

    pub fn lock(&self) -> RegistryGuard {
        RegistryGuard(self.0.lock().unwrap())
    }
}

pub struct RegistryGuard<'a>(MutexGuard<'a, libworldgen::registry::Registries>);

impl<'a> Deref for RegistryGuard<'a> {
    type Target = libworldgen::registry::Registries;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> DerefMut for RegistryGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> AsRef<Registries> for RegistryGuard<'a> {
    fn as_ref(&self) -> &Registries {
        (self.0.loader.as_ref() as &(dyn Any + Send))
            .downcast_ref()
            .unwrap()
    }
}

impl<'a> AsMut<Registries> for RegistryGuard<'a> {
    fn as_mut(&mut self) -> &mut Registries {
        (self.0.loader.as_mut() as &mut (dyn Any + Send))
            .downcast_mut()
            .unwrap()
    }
}

#[derive(Component)]
pub struct ChunkPos(pub IVec2);

#[derive(Component)]
pub struct ChunkTask {
    task: Option<Task<Mesh>>,
    cancelled: bool,
}

impl ChunkTask {
    pub fn create(registries: RegistriesHandle, pos: IVec2) -> Self {
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let chunk = Box::new(Chunk::generate_test_chunk(registries, pos));
            let culled_chunk = CulledChunk::new(chunk);
            culled_chunk.to_mesh().unwrap().to_bevy_mesh()
        });

        Self {
            task: Some(task),
            cancelled: false,
        }
    }

    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn try_complete(
        &mut self,
        self_entity: Entity,
        commands: &mut Commands,
        pos: &IVec2,
        atlas: &mut ResMut<DynamicTextureAtlas>,
        meshes: &mut ResMut<Assets<Mesh>>,
    ) {
        if self.cancelled {
            commands.entity(self_entity).despawn();
            return;
        }
        if self.task.as_ref().unwrap().is_finished() {
            let mesh = block_on(self.task.take().unwrap());
            commands
                .entity(self_entity)
                .insert((
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(atlas.material.clone()),
                    Transform::from_xyz(pos.x as f32 * 16.0, -64.0, pos.y as f32 * 16.0),
                ))
                .remove::<ChunkTask>();
        }
    }

    pub fn init_sys(
        mut commands: Commands,
        mut next_state: ResMut<NextState<AppState>>,
        asset_server: Res<AssetServer>,
    ) {
        let packs = get_packs();

        if let Err(err) = packs {
            error!("Unable to load resource packs: {}", err);
            commands.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(0.0),
                    bottom: Val::Px(0.0),
                    left: Val::Px(0.0),
                    right: Val::Px(0.0),
                    display: Display::Grid,
                    justify_items: JustifyItems::Center,
                    align_items: AlignItems::Center,
                    ..Default::default()
                },
                children![(
                    Text::new("Unable to load resource packs.\nSee logs for more info."),
                    TextColor(Color::srgb_u8(255, 152, 148)),
                    TextFont {
                        font: asset_server.load("JetBrainsMono-Regular.ttf"),
                        font_size: 40.0,
                        ..Default::default()
                    }
                )],
            ));
            return;
        }
        let packs = packs.unwrap();

        let (from_loader_send, from_loader_recv) = crossbeam_channel::unbounded();
        commands.insert_resource(LoaderChannels {
            from_loader: from_loader_recv,
        });
        commands.insert_resource(RegistriesHandle::new(packs, from_loader_send));
        next_state.set(AppState::Main);
    }

    pub fn try_complete_sys(
        mut commands: Commands,
        mut atlas: ResMut<DynamicTextureAtlas>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut query: Query<(Entity, &mut ChunkTask, &ChunkPos)>,
    ) {
        for (entity, mut task, pos) in &mut query {
            task.try_complete(entity, &mut commands, &pos.0, &mut atlas, &mut meshes);
        }
    }

    pub fn load_textures_sys(
        mut atlas: ResMut<DynamicTextureAtlas>,
        mut layouts: ResMut<Assets<TextureAtlasLayout>>,
        mut images: ResMut<Assets<Image>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        channels: ResMut<LoaderChannels>,
    ) {
        while let Ok(msg) = channels.from_loader.try_recv() {
            match msg {
                FromLoader::LoadTextureIntoAtlas(texture) => {
                    let _span = info_span!("LoadTextureIntoAtlas").entered();
                    let image = Image::from_dynamic(
                        DynamicImage::from(*texture),
                        true,
                        RenderAssetUsages::MAIN_WORLD,
                    );
                    atlas
                        .add(&image, layouts.as_mut(), images.as_mut())
                        .unwrap();

                    // make bevy update texture on the GPU
                    materials.get_mut(&atlas.material);
                }
            }
        }
    }
}

pub struct Registries {
    pub blockstates: Registry<BlockstateFile>,
    pub models: Registry<LoadedBlockModel>,
    pub textures: Registry<LoadedTexture>,
    pub atlas_alloc: AtlasAllocator,
    pub from_loader_send: Sender<FromLoader>,
    packs: Vec<PathBuf>,
    archives: HashMap<String, ZipArchive<File>>,
}

impl Registries {
    pub fn new(packs: Vec<PathBuf>, from_loader_send: Sender<FromLoader>) -> Self {
        Self {
            blockstates: Registry::new(),
            models: Registry::new(),
            textures: Registry::new(),
            atlas_alloc: AtlasAllocator::new(Size2D::new(ATLAS_SIZE as i32, ATLAS_SIZE as i32)),
            from_loader_send,
            packs,
            archives: HashMap::new(),
        }
    }

    pub fn get<'a, L: RegistryResource + 'a>(
        &'a self,
        res_loc: &ResourceLocation<L::Resource>,
    ) -> Option<Result<&'a L, &'a ResourceParseError>> {
        L::get(self).loaded.get(res_loc).map(|v| v.as_ref())
    }

    fn get_or_insert_with<'a, L: RegistryResource + 'a, F>(
        &'a mut self,
        key: &ResourceLocation<L::Resource>,
        f: F,
    ) -> Result<&'a L, &'a ResourceParseError>
    where
        F: FnOnce(&mut Self) -> Result<L, ResourceParseError>,
    {
        if L::get(self).loaded.contains_key(key) {
            L::get(self).loaded[key].as_ref()
        } else if L::get(self).loading.contains(key) {
            block_on(poll_fn(|_| {
                if let Some(val) = L::get(self).loaded.get(key) {
                    Poll::Ready(val.as_ref())
                } else {
                    Poll::Pending
                }
            }))
        } else {
            L::get_mut(self).loading.insert(key.clone());
            let res = f(self);
            L::get_mut(self).loaded.insert(key.clone(), res);
            L::get_mut(self).loading.remove(key);
            L::get(self).loaded[key].as_ref()
        }
    }

    pub fn get_or_load<'a, L: RegistryResource + 'a>(
        &'a mut self,
        res_loc: &ResourceLocation<L::Resource>,
    ) -> Result<&'a L, &'a ResourceParseError> {
        Registry::get_or_load(self, res_loc)
    }

    pub fn get_total_status(&self) -> Status {
        let textures = self.textures.get_status();
        let blockstates = self.blockstates.get_status();
        let models = self.models.get_status();
        Status {
            ok: textures.ok + blockstates.ok + models.ok,
            errs: textures.errs + blockstates.errs + models.errs,
            loading: textures.loading + blockstates.loading + models.loading,
        }
    }
}

impl ResourceLoader for Registries {
    fn load_noise_parameters(
        &mut self,
        res_loc: &ResourceLocation<NoiseParameters>,
    ) -> Result<NoiseParameters, libworldgen::error::Error> {
        Registry::load(self, res_loc).map_err(|e| e.into())
    }

    fn load_density_function(
        &mut self,
        res_loc: &ResourceLocation<DensityFunction>,
    ) -> Result<DensityFunction, libworldgen::error::Error> {
        Registry::load(self, res_loc).map_err(|e| e.into())
    }
}

fn get_packs() -> Result<Vec<PathBuf>, ResourceParseError> {
    let mut paths = vec![];

    for path in env::args().skip(1) {
        let path_buf = PathBuf::from_str(&path).expect("infallible");
        if let Some(ext) = path_buf.extension() {
            if !(ext == OsStr::new("zip") || ext == OsStr::new("jar")) {
                return Err(
                    "Unsupported resource pack type. Expected `.zip`, `.jar` or a plain directory"
                        .into(),
                );
            }
        }

        paths.push(path_buf);
    }

    if paths.is_empty() {
        Err("No resource packs added. See README.md".into())
    } else {
        Ok(paths)
    }
}

/// Extracts a single file from a zip archive to some opaque location.
/// Returns the file path of the extracted file, if it exists.
#[instrument(skip_all)]
fn extract_file(
    archive_path: &Path,
    file_path: &Path,
    archives: &mut HashMap<String, ZipArchive<File>>,
) -> Option<PathBuf> {
    let archive = archives
        .entry(archive_path.to_str().unwrap().to_string())
        .or_insert_with(|| {
            let zipfile = std::fs::File::open(archive_path).unwrap();
            zip::ZipArchive::new(zipfile).unwrap()
        });
    let mut file = archive
        .by_name(&file_path.to_str().unwrap().replace('\\', "/"))
        .ok()?;

    let mut outpath = PathBuf::new();
    outpath.push("worldgenpreview_cache");
    let mut fhash = FixedHasher.build_hasher();
    archive_path.hash(&mut fhash);
    file_path.hash(&mut fhash);
    outpath.push(format!("{:x}", fhash.finish()));

    if let Some(p) = outpath.parent() {
        if !p.exists() {
            fs::create_dir_all(p).unwrap();
        }
    }
    let mut outfile = fs::File::create(&outpath).unwrap();
    io::copy(&mut file, &mut outfile).unwrap();

    Some(outpath)
}

pub struct Registry<L: LoadedResource> {
    loaded: HashMap<ResourceLocation<L::Resource>, Result<L, ResourceParseError>>,
    loading: HashSet<ResourceLocation<L::Resource>>,
}

impl<L: LoadedResource> Registry<L> {
    fn load(
        registries: &mut Registries,
        res_loc: &ResourceLocation<L::Resource>,
    ) -> Result<L, ResourceParseError> {
        let mut resource = None;
        for pack in &registries.packs {
            if let Some(ext) = pack.extension() {
                if ext == OsStr::new("zip") || ext == OsStr::new("jar") {
                    let path = extract_file(
                        pack,
                        &L::Resource::to_path(res_loc),
                        &mut registries.archives,
                    );
                    if let Some(path) = path {
                        resource = Some(L::Resource::open(path)?);
                        break;
                    }
                }
            }
            let mut path = pack.clone();
            path.push(L::Resource::to_path(res_loc));
            if fs::exists(&path)? {
                resource = Some(L::Resource::open(path)?);
                break;
            }
        }

        if let Some(model) = resource {
            L::load(res_loc, model, registries)
        } else {
            Err(ResourceParseError::ResourceNotFound(
                registries
                    .packs
                    .iter()
                    .map(|pack| {
                        let mut path = pack.clone();
                        path.push(L::Resource::to_path(res_loc));
                        path.to_string_lossy().to_string()
                    })
                    .collect(),
            ))
        }
    }
}

impl<L: RegistryResource> Registry<L> {
    pub fn new() -> Self {
        Self {
            loaded: HashMap::new(),
            loading: HashSet::new(),
        }
    }

    /// Gets a resource from the registry if it exists or loads it into the registry if it doesn't exist yet.
    pub fn get_or_load<'a>(
        registries: &'a mut Registries,
        res_loc: &ResourceLocation<L::Resource>,
    ) -> Result<&'a L, &'a ResourceParseError>
    where
        L: 'a,
    {
        registries.get_or_insert_with(res_loc, |registries| {
            let res = Self::load(registries, res_loc);
            if let Err(err) = &res {
                error!(
                    "Failed to load resource type {}: {}.\nError: {}",
                    type_name::<L::Resource>(),
                    res_loc,
                    err
                );
            }
            res
        })
    }

    pub fn get_status(&self) -> Status {
        let ok = self.loaded.iter().filter(|v| v.1.is_ok()).count();
        Status {
            ok,
            errs: self.loaded.len() - ok,
            loading: self.loading.len(),
        }
    }
}

pub enum FromLoader {
    LoadTextureIntoAtlas(Box<RgbaImage>),
}

#[derive(Resource)]
pub struct LoaderChannels {
    pub from_loader: Receiver<FromLoader>,
}

pub struct Status {
    pub ok: usize,
    pub errs: usize,
    pub loading: usize,
}

pub trait RegistryResource: LoadedResource {
    fn get(registries: &Registries) -> &Registry<Self>;

    fn get_mut(registries: &mut Registries) -> &mut Registry<Self>;
}

pub trait LoadedResource: Sized {
    type Resource: ResourceType;

    fn load(
        res_loc: &ResourceLocation<Self::Resource>,
        res: Self::Resource,
        registries: &mut Registries,
    ) -> Result<Self, ResourceParseError>;
}

impl<T: ResourceType> LoadedResource for T {
    type Resource = Self;
    fn load(
        _res_loc: &ResourceLocation<Self::Resource>,
        res: Self::Resource,
        _registries: &mut Registries,
    ) -> Result<Self, ResourceParseError> {
        Ok(res)
    }
}

impl RegistryResource for BlockstateFile {
    fn get(registries: &Registries) -> &Registry<Self> {
        &registries.blockstates
    }

    fn get_mut(registries: &mut Registries) -> &mut Registry<Self> {
        &mut registries.blockstates
    }
}
