use std::{fs, future::poll_fn, path::PathBuf, str::FromStr, task::Poll};

use async_channel::{Receiver, Sender};
use bevy::{
    asset::{Assets, RenderAssetUsages},
    ecs::{
        resource::Resource,
        system::{Commands, ResMut},
    },
    image::{Image, TextureAtlasLayout},
    math::IVec2,
    pbr::MeshMaterial3d,
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
use tracing::info_span;

use crate::{
    chunk::{Chunk, CulledChunk, LoadedBlockModel},
    textures::{ATLAS_SIZE, DynamicTextureAtlas, LoadedTexture},
};
pub struct Registries {
    pub blockstates: Registry<BlockstateFile>,
    pub models: Registry<LoadedBlockModel>,
    pub textures: Registry<LoadedTexture>,
    pub atlas_alloc: AtlasAllocator,
    pub to_loader_recv: Receiver<ToLoader>,
    pub from_loader_send: Sender<FromLoader>,
    packs: Vec<PathBuf>,
}

impl Registries {
    pub fn new(
        packs: Vec<PathBuf>,
        to_loader_recv: Receiver<ToLoader>,
        from_loader_send: Sender<FromLoader>,
    ) -> Self {
        Self {
            blockstates: Registry::new(),
            models: Registry::new(),
            textures: Registry::new(),
            atlas_alloc: AtlasAllocator::new(Size2D::new(ATLAS_SIZE as i32, ATLAS_SIZE as i32)),
            to_loader_recv,
            from_loader_send,
            packs,
        }
    }

    pub fn init_sys(mut commands: Commands) {
        let (to_loader_send, to_loader_recv) = async_channel::unbounded();
        let (from_loader_send, from_loader_recv) = async_channel::unbounded();
        commands.insert_resource(LoaderChannels {
            to_loader: to_loader_send,
            from_loader: from_loader_recv,
        });

        AsyncComputeTaskPool::get()
            .spawn(async move {
                let mut reg = Registries::new(
                    vec![PathBuf::from_str("../datapacks/minecraft").unwrap()],
                    to_loader_recv,
                    from_loader_send,
                );

                while let Ok(msg) = reg.to_loader_recv.recv().await {
                    match msg {
                        ToLoader::GenChunk(pos) => {
                            let chunk = Box::new(Chunk::example_chunk(&mut reg).await);
                            let culled_chunk = CulledChunk::new(chunk);
                            let chunk_mesh = culled_chunk.to_mesh().unwrap().to_bevy_mesh();
                            reg.from_loader_send
                                .send(FromLoader::LoadedChunk(pos, chunk_mesh))
                                .await
                                .unwrap();
                        } //     ToLoader::LoadBlockModel(model) => {
                          //         let res: LoadedBlockModel = reg
                          //             .get_or_load::<LoadedBlockModel>(&model)
                          //             .await
                          //             .unwrap()
                          //             .clone();
                          //         let mesh = res
                          //             .mesh
                          //             .unwrap()
                          //             .to_mesh(
                          //                 HashSet::from_iter(vec![
                          //                     Direction::Down,
                          //                     Direction::Up,
                          //                     Direction::North,
                          //                     Direction::South,
                          //                     Direction::East,
                          //                     Direction::West,
                          //                 ]),
                          //                 &res.textures,
                          //                 &reg,
                          //             )
                          //             .unwrap();
                          //         reg.from_loader_send
                          //             .send(FromLoader::LoadedBlockModel(model, Box::new(mesh)))
                          //             .await
                          //             .unwrap();
                          //     }
                    }
                }
            })
            .detach();
    }

    pub fn update_sys(
        mut commands: Commands,
        channels: ResMut<LoaderChannels>,
        mut atlas: ResMut<DynamicTextureAtlas>,
        mut layouts: ResMut<Assets<TextureAtlasLayout>>,
        mut images: ResMut<Assets<Image>>,
        mut meshes: ResMut<Assets<Mesh>>,
    ) {
        while let Ok(msg) = channels.from_loader.try_recv() {
            match msg {
                FromLoader::LoadedChunk(pos, mesh) => {
                    let _ = info_span!("LoadedChunk").entered();
                    images
                        .get(&atlas.atlas)
                        .unwrap()
                        .clone()
                        .try_into_dynamic()
                        .unwrap()
                        .save("atlas.png")
                        .unwrap();
                    commands.spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(atlas.material.clone()),
                        Transform::from_xyz(pos.x as f32 * 16.0, -64.0, pos.y as f32 * 16.0),
                        // Transform::from_xyz(0.0, -64.0, 0.0),
                    ));
                }
                FromLoader::LoadTextureIntoAtlas(texture) => {
                    let _ = info_span!("LoadTextureIntoAtlas").entered();
                    let image = Image::from_dynamic(
                        DynamicImage::from(*texture),
                        true,
                        RenderAssetUsages::MAIN_WORLD,
                    );
                    atlas
                        .add(&image, layouts.as_mut(), images.as_mut())
                        .unwrap();
                    // images
                    //     .get(&atlas.atlas)
                    //     .unwrap()
                    //     .clone()
                    //     .try_into_dynamic()
                    //     .unwrap()
                    //     .save("atlas.png")
                    //     .unwrap();
                } //     FromLoader::LoadedBlockModel(_res_loc, mesh) => {
                  //         commands.spawn((
                  //             Mesh3d(meshes.add(*mesh)),
                  //             MeshMaterial3d(atlas.material.clone()),
                  //             Transform::from_xyz(3.0, 0.0, *i * 1.5),
                  //         ));
                  //         *i += 1.0;
                  //     }
            }
        }
    }

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
        let mut model = None;
        for pack in &registries.packs {
            let mut path = pack.clone();
            path.push(L::Resource::to_path(res_loc));
            if fs::exists(&path)? {
                model = Some(L::Resource::open(path)?);
                break;
            }
        }

        if let Some(model) = model {
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
                Self::load(registries, res_loc).await
            })
            .await
    }
}

pub enum ToLoader {
    GenChunk(IVec2),
}

pub enum FromLoader {
    LoadedChunk(IVec2, Mesh),
    LoadTextureIntoAtlas(Box<RgbaImage>),
}

#[derive(Resource)]
pub struct LoaderChannels {
    pub to_loader: Sender<ToLoader>,
    pub from_loader: Receiver<FromLoader>,
}

pub struct Status {
    ok: usize,
    errs: usize,
    loading: usize,
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
