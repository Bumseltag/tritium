use std::{fs, marker::PhantomData, path::PathBuf};

use hashbrown::{HashMap, hash_map::EntryRef};

use crate::{ResourceLocation, ResourceParseError, ResourceType};

#[derive(Clone)]
pub struct Registry<T: ResourceType, L: ResourceProcessor<T> = NoopLoader> {
    map: HashMap<ResourceLocation<T>, L::Loaded>,
    packs: Vec<PathBuf>,
    _marker: PhantomData<L>,
}

impl<T: ResourceType, L: ResourceProcessor<T>> Registry<T, L> {
    pub fn new(pack_locations: Vec<PathBuf>) -> Self {
        Self {
            map: HashMap::new(),
            packs: pack_locations,
            _marker: PhantomData,
        }
    }

    pub fn get_or_load<'a>(
        &mut self,
        res_loc: &ResourceLocation<T>,
        state: L::State<'a>,
    ) -> Result<&L::Loaded, ResourceParseError> {
        get_or_insert_with(&mut self.map, res_loc, |res_loc| {
            let mut val = None;
            for pack in &self.packs {
                let mut path = pack.clone();
                path.push(T::to_path(res_loc));
                if fs::exists(&path)? {
                    val = Some(L::load(res_loc, T::open(path)?, state));
                    break;
                }
            }
            if let Some(val) = val {
                Ok(val)
            } else {
                Err(ResourceParseError::ResourceNotFound(
                    self.packs
                        .iter()
                        .map(|pack| {
                            let mut path = pack.clone();
                            path.push(T::to_path(res_loc));
                            path.to_string_lossy().to_string()
                        })
                        .collect(),
                ))
            }
        })
    }
}

pub trait ResourceProcessor<T: ResourceType> {
    type Loaded;
    type State<'a>;

    fn load<'a>(res_loc: &ResourceLocation<T>, resource: T, state: Self::State<'a>)
    -> Self::Loaded;
}

pub struct NoopLoader;

impl<T: ResourceType> ResourceProcessor<T> for NoopLoader {
    type Loaded = T;
    type State<'a> = ();

    fn load<'a>(
        _res_loc: &ResourceLocation<T>,
        resource: T,
        _state: Self::State<'a>,
    ) -> Self::Loaded {
        resource
    }
}

pub fn get_or_insert_with<'a, T: ResourceType, L, F>(
    map: &'a mut HashMap<ResourceLocation<T>, L>,
    res_loc: &ResourceLocation<T>,
    f: F,
) -> Result<&'a L, ResourceParseError>
where
    F: FnOnce(&ResourceLocation<T>) -> Result<L, ResourceParseError>,
{
    match map.entry_ref(res_loc) {
        EntryRef::Occupied(occ) => Ok(occ.into_mut()),
        EntryRef::Vacant(vac) => Ok(vac.insert(f(res_loc)?)),
    }
}
