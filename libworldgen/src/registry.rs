use std::{any::Any, sync::Arc};

use hashbrown::HashMap;
use mcpackloader::{
    ResourceLocation,
    worldgen::{DensityFunction, NoiseParameters},
};

use crate::{density_function::DynFnOpType, error::Error};

pub struct Registries {
    noise_params: MutRegistry<NoiseParameters>,
    density_function: MutRegistry<DensityFunction>,
    density_function_ops: MutRegistry<DynFnOpType>,
    pub loader: Box<dyn ResourceLoader + Send + Sync>,
}

impl Registries {
    pub fn new(loader: Box<dyn ResourceLoader + Send + Sync>) -> Self {
        Self {
            noise_params: MutRegistry::new(),
            density_function: MutRegistry::new(),
            density_function_ops: MutRegistry::new(),
            loader,
        }
    }

    pub fn get<T: RegistryType>(&self, res_loc: &ResourceLocation<T>) -> Option<Arc<T>> {
        T::get(self).get(res_loc)
    }

    pub fn get_result<T: RegistryType>(
        &self,
        res_loc: &ResourceLocation<T>,
    ) -> Option<Result<Arc<T>, Arc<Error>>> {
        T::get(self).get_result(res_loc)
    }

    pub fn get_or_load<T: MutRegistryType>(
        &mut self,
        res_loc: &ResourceLocation<T>,
    ) -> Result<Arc<T>, Arc<Error>> {
        MutRegistry::get_or_load(self, res_loc)
    }

    #[track_caller]
    pub fn insert<T: RegistryType>(&mut self, res_loc: ResourceLocation<T>, value: T) {
        T::get_mut(self).insert(res_loc, value);
    }
}

pub struct MutRegistry<T: RegistryType> {
    loaded: HashMap<ResourceLocation<T>, Arc<T>>,
    errors: HashMap<ResourceLocation<T>, Arc<Error>>,
}

impl<T: RegistryType> MutRegistry<T> {
    pub fn new() -> Self {
        Self {
            loaded: HashMap::new(),
            errors: HashMap::new(),
        }
    }

    fn get(&self, res_loc: &ResourceLocation<T>) -> Option<Arc<T>> {
        self.loaded.get(res_loc).cloned()
    }

    fn get_result(&self, res_loc: &ResourceLocation<T>) -> Option<Result<Arc<T>, Arc<Error>>> {
        if let Some(loaded) = self.loaded.get(res_loc) {
            Some(Ok(loaded.clone()))
        } else {
            self.errors.get(res_loc).map(|error| Err(error.clone()))
        }
    }

    #[track_caller]
    fn insert(&mut self, res_loc: ResourceLocation<T>, value: T) {
        if self.errors.contains_key(&res_loc) || self.loaded.contains_key(&res_loc) {
            panic!("{res_loc} already exists in this registry");
        } else {
            self.loaded.insert(res_loc, Arc::new(value));
        }
    }
}

impl<T: MutRegistryType> MutRegistry<T> {
    fn get_or_load(
        reg: &mut Registries,
        res_loc: &ResourceLocation<T>,
    ) -> Result<Arc<T>, Arc<Error>> {
        if let Some(res) = T::get(reg).get(res_loc) {
            Ok(res)
        } else {
            let loaded = T::load(reg.loader.as_mut(), res_loc)
                .map(Arc::new)
                .map_err(Arc::new);
            match loaded {
                Ok(loaded) => Ok(T::get_mut(reg)
                    .loaded
                    .entry_ref(res_loc)
                    .insert(loaded)
                    .get()
                    .clone()),
                Err(error) => Err(T::get_mut(reg)
                    .errors
                    .entry_ref(res_loc)
                    .insert(error)
                    .get()
                    .clone()),
            }
        }
    }
}

impl<T: RegistryType> Default for MutRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub trait ResourceLoader: Any {
    fn load_noise_parameters(
        &mut self,
        res_loc: &ResourceLocation<NoiseParameters>,
    ) -> Result<NoiseParameters, Error>;
    fn load_density_function(
        &mut self,
        res_loc: &ResourceLocation<DensityFunction>,
    ) -> Result<DensityFunction, Error>;
}

pub trait MutRegistryType: RegistryType + Sized {
    fn load(
        loader: &mut dyn ResourceLoader,
        res_loc: &ResourceLocation<Self>,
    ) -> Result<Self, Error>;
}

pub trait RegistryType: Sized {
    fn get(reg: &Registries) -> &MutRegistry<Self>;
    fn get_mut(reg: &mut Registries) -> &mut MutRegistry<Self>;
}

impl MutRegistryType for NoiseParameters {
    fn load(
        loader: &mut dyn ResourceLoader,
        res_loc: &ResourceLocation<Self>,
    ) -> Result<Self, Error> {
        loader.load_noise_parameters(res_loc)
    }
}

impl RegistryType for NoiseParameters {
    fn get(reg: &Registries) -> &MutRegistry<Self> {
        &reg.noise_params
    }

    fn get_mut(reg: &mut Registries) -> &mut MutRegistry<Self> {
        &mut reg.noise_params
    }
}

impl MutRegistryType for DensityFunction {
    fn load(
        loader: &mut dyn ResourceLoader,
        res_loc: &ResourceLocation<Self>,
    ) -> Result<Self, Error> {
        loader.load_density_function(res_loc)
    }
}

impl RegistryType for DensityFunction {
    fn get(reg: &Registries) -> &MutRegistry<Self> {
        &reg.density_function
    }

    fn get_mut(reg: &mut Registries) -> &mut MutRegistry<Self> {
        &mut reg.density_function
    }
}

impl RegistryType for DynFnOpType {
    fn get(reg: &Registries) -> &MutRegistry<Self> {
        &reg.density_function_ops
    }

    fn get_mut(reg: &mut Registries) -> &mut MutRegistry<Self> {
        &mut reg.density_function_ops
    }
}
