use std::ffi::OsStr;
use std::fs::File;
use std::hash::{BuildHasher, Hash, Hasher};
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::{any::type_name, env, fs, future::poll_fn, path::PathBuf, str::FromStr, task::Poll};

use async_channel::{Receiver, Sender};
use async_lock::{Mutex, MutexGuardArc};
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
use guillotiere::{AtlasAllocator, euclid::Size2D};
use image::{DynamicImage, RgbaImage};
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
pub struct RegistriesHandle(Arc<Mutex<Registries>>);

impl RegistriesHandle {
    pub async fn lock(&self) -> MutexGuardArc<Registries> {
        self.0.lock_arc().await
    }

    pub fn lock_blocking(&self) -> MutexGuardArc<Registries> {
        self.0.lock_arc_blocking()
    }
}

#[derive(Component)]
pub struct ChunkTask {
    pos: IVec2,
    task: Option<Task<Mesh>>,
}

impl ChunkTask {
    pub fn create(pos: IVec2, registries: RegistriesHandle) -> Self {
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let chunk = Box::new(Chunk::example_chunk(&mut *registries.lock().await).await);
            let culled_chunk = CulledChunk::new(chunk);
            culled_chunk.to_mesh().unwrap().to_bevy_mesh()
        });

        Self {
            pos,
            task: Some(task),
        }
    }

    fn try_complete(
        &mut self,
        self_entity: Entity,
        commands: &mut Commands,
        atlas: &mut ResMut<DynamicTextureAtlas>,
        meshes: &mut ResMut<Assets<Mesh>>,
    ) {
        if self.task.as_ref().unwrap().is_finished() {
            let mesh = block_on(self.task.take().unwrap());
            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(atlas.material.clone()),
                Transform::from_xyz(self.pos.x as f32 * 16.0, -64.0, self.pos.y as f32 * 16.0),
            ));
            commands.entity(self_entity).despawn();
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

        let (from_loader_send, from_loader_recv) = async_channel::unbounded();
        commands.insert_resource(LoaderChannels {
            from_loader: from_loader_recv,
        });
        commands.insert_resource(RegistriesHandle(Arc::new(Mutex::new(Registries::new(
            packs,
            from_loader_send,
        )))));
        next_state.set(AppState::Main);
    }

    pub fn try_complete_sys(
        mut commands: Commands,
        mut atlas: ResMut<DynamicTextureAtlas>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut query: Query<(Entity, &mut ChunkTask)>,
    ) {
        for (entity, mut task) in &mut query {
            task.try_complete(entity, &mut commands, &mut atlas, &mut meshes);
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

                    // make bevy update canvas on the GPU
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

    // pub fn init_sys(
    //     mut commands: Commands,
    //     mut next_state: ResMut<NextState<AppState>>,
    //     asset_server: Res<AssetServer>,
    // ) {
    //     let packs = get_packs();

    //     if let Err(err) = packs {
    //         error!("Unable to load resource packs: {}", err);
    //         commands.spawn((
    //             Node {
    //                 position_type: PositionType::Absolute,
    //                 top: Val::Px(0.0),
    //                 bottom: Val::Px(0.0),
    //                 left: Val::Px(0.0),
    //                 right: Val::Px(0.0),
    //                 display: Display::Grid,
    //                 justify_items: JustifyItems::Center,
    //                 align_items: AlignItems::Center,
    //                 ..Default::default()
    //             },
    //             children![(
    //                 Text::new("Unable to load resource packs.\nSee logs for more info."),
    //                 TextColor(Color::srgb_u8(255, 152, 148)),
    //                 TextFont {
    //                     font: asset_server.load("JetBrainsMono-Regular.ttf"),
    //                     font_size: 40.0,
    //                     ..Default::default()
    //                 }
    //             )],
    //         ));
    //         return;
    //     }
    //     let packs = packs.unwrap();

    //     let (to_loader_send, to_loader_recv) = async_channel::unbounded();
    //     let (from_loader_send, from_loader_recv) = async_channel::unbounded();
    //     commands.insert_resource(LoaderChannels {
    //         to_loader: to_loader_send,
    //         from_loader: from_loader_recv,
    //     });

    //     AsyncComputeTaskPool::get()
    //         .spawn(async move {
    //             let mut reg = Registries::new(packs, to_loader_recv, from_loader_send);

    //             while let Ok(msg) = reg.to_loader_recv.recv().await {
    //                 match msg {
    //                     ToLoader::GenChunk(pos) => {
    //                         let chunk = Box::new(Chunk::example_chunk(&mut reg).await);
    //                         let culled_chunk = CulledChunk::new(chunk);
    //                         let chunk_mesh = culled_chunk.to_mesh().unwrap().to_bevy_mesh();
    //                         reg.from_loader_send
    //                             .send(FromLoader::LoadedChunk(
    //                                 pos,
    //                                 chunk_mesh,
    //                                 reg.get_total_status(),
    //                             ))
    //                             .await
    //                             .unwrap();
    //                     }
    //                 }
    //             }
    //         })
    //         .detach();

    //     next_state.set(AppState::Main);
    // }

    // #[allow(clippy::too_many_arguments)]
    // pub fn update_sys(
    //     mut commands: Commands,
    //     channels: ResMut<LoaderChannels>,
    //     mut atlas: ResMut<DynamicTextureAtlas>,
    //     mut layouts: ResMut<Assets<TextureAtlasLayout>>,
    //     mut images: ResMut<Assets<Image>>,
    //     mut meshes: ResMut<Assets<Mesh>>,
    //     mut status_res: ResMut<RegistryStatus>,
    //     mut materials: ResMut<Assets<StandardMaterial>>,
    // ) {
    //     while let Ok(msg) = channels.from_loader.try_recv() {
    //         match msg {
    //             FromLoader::LoadedChunk(pos, mesh, status) => {
    //                 let _span = info_span!("LoadedChunk").entered();
    //                 // images
    //                 //     .get(&atlas.atlas)
    //                 //     .unwrap()
    //                 //     .clone()
    //                 //     .try_into_dynamic()
    //                 //     .unwrap()
    //                 //     .save("atlas.png")
    //                 //     .unwrap();
    //                 commands.spawn((
    //                     Mesh3d(meshes.add(mesh)),
    //                     MeshMaterial3d(atlas.material.clone()),
    //                     Transform::from_xyz(pos.x as f32 * 16.0, -64.0, pos.y as f32 * 16.0),
    //                     // Transform::from_xyz(0.0, -64.0, 0.0),
    //                 ));
    //                 status_res.0 = status;
    //             }
    //             FromLoader::LoadTextureIntoAtlas(texture) => {
    //                 let _span = info_span!("LoadTextureIntoAtlas").entered();
    //                 let image = Image::from_dynamic(
    //                     DynamicImage::from(*texture),
    //                     true,
    //                     RenderAssetUsages::MAIN_WORLD,
    //                 );
    //                 atlas
    //                     .add(&image, layouts.as_mut(), images.as_mut())
    //                     .unwrap();

    //                 // make bevy update canvas on the GPU
    //                 materials.get_mut(&atlas.material);

    //                 // images
    //                 //     .get(&atlas.atlas)
    //                 //     .unwrap()
    //                 //     .clone()
    //                 //     .try_into_dynamic()
    //                 //     .unwrap()
    //                 //     .save("atlas.png")
    //                 //     .unwrap();
    //             } //     FromLoader::LoadedBlockModel(_res_loc, mesh) => {
    //               //         commands.spawn((
    //               //             Mesh3d(meshes.add(*mesh)),
    //               //             MeshMaterial3d(atlas.material.clone()),
    //               //             Transform::from_xyz(3.0, 0.0, *i * 1.5),
    //               //         ));
    //               //         *i += 1.0;
    //               //     }
    //         }
    //     }
    // }

    pub fn get<'a, L: LoadedResource + 'a>(
        &'a self,
        res_loc: &ResourceLocation<L::Resource>,
    ) -> Option<Result<&'a L, &'a ResourceParseError>> {
        L::get(self).loaded.get(res_loc).map(|v| v.as_ref())
    }

    async fn get_or_insert_with<'a, L: LoadedResource + 'a, F>(
        &'a mut self,
        key: &ResourceLocation<L::Resource>,
        f: F,
    ) -> Result<&'a L, &'a ResourceParseError>
    where
        F: AsyncFnOnce(&mut Self) -> Result<L, ResourceParseError>,
    {
        if L::get(self).loaded.contains_key(key) {
            L::get(self).loaded[key].as_ref()
        } else if L::get(self).loading.contains(key) {
            poll_fn(|_| {
                if let Some(val) = L::get(self).loaded.get(key) {
                    Poll::Ready(val.as_ref())
                } else {
                    Poll::Pending
                }
            })
            .await
        } else {
            L::get_mut(self).loading.insert(key.clone());
            let res = f(self).await;
            L::get_mut(self).loaded.insert(key.clone(), res);
            L::get_mut(self).loading.remove(key);
            L::get(self).loaded[key].as_ref()
        }
    }

    pub async fn get_or_load<'a, L: LoadedResource + 'a>(
        &'a mut self,
        res_loc: &ResourceLocation<L::Resource>,
    ) -> Result<&'a L, &'a ResourceParseError> {
        Box::pin(Registry::get_or_load(self, res_loc)).await
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
    pub fn new() -> Self {
        Self {
            loaded: HashMap::new(),
            loading: HashSet::new(),
        }
    }

    async fn load(
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
            L::load(res_loc, model, registries).await
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

    /// Gets a resource from the registry if it exists or loads it into the registry if it doesn't exist yet.
    pub async fn get_or_load<'a>(
        registries: &'a mut Registries,
        res_loc: &ResourceLocation<L::Resource>,
    ) -> Result<&'a L, &'a ResourceParseError>
    where
        L: 'a,
    {
        registries
            .get_or_insert_with(res_loc, async |registries| {
                let res = Self::load(registries, res_loc).await;
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
            .await
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

pub trait LoadedResource: Sized {
    type Resource: ResourceType;

    async fn load(
        res_loc: &ResourceLocation<Self::Resource>,
        res: Self::Resource,
        registries: &mut Registries,
    ) -> Result<Self, ResourceParseError>;

    fn get(registries: &Registries) -> &Registry<Self>;

    fn get_mut(registries: &mut Registries) -> &mut Registry<Self>;
}

impl LoadedResource for BlockstateFile {
    type Resource = Self;

    async fn load(
        _res_loc: &ResourceLocation<Self::Resource>,
        res: Self::Resource,
        _registries: &mut Registries,
    ) -> Result<Self, ResourceParseError> {
        Ok(res)
    }

    fn get(registries: &Registries) -> &Registry<Self> {
        &registries.blockstates
    }

    fn get_mut(registries: &mut Registries) -> &mut Registry<Self> {
        &mut registries.blockstates
    }
}
