use std::marker::PhantomData;
use std::{str::FromStr, sync::Arc};

use glam::I64Vec3;
use mcpackloader::{ResourceLocation, worldgen::DensityFunction};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{error::Error, registry::Registries};

use crate::density_function::std_ops::Constant;

pub mod std_ops;

pub const SUBCHUNK_SIZE: usize = 16 * 16 * 16;

pub trait FunctionOp: Send + Sync {
    fn run_once(&self, pos: &I64Vec3) -> f64;
    fn run_plane(&self, pos: &I64Vec3) -> [f64; 16 * 16] {
        let mut res = [f64::NAN; 16 * 16];
        for x in 0..16 {
            for z in 0..16 {
                res[x + (z * 16)] = self.run_once(&(pos + I64Vec3::new(x as i64, 0, z as i64)))
            }
        }
        res
    }

    fn run_subchunk(&self, pos: &I64Vec3) -> [f64; SUBCHUNK_SIZE] {
        let mut res = [f64::NAN; SUBCHUNK_SIZE];
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    res[x + (z * 16) + (y * 16 * 16)] =
                        self.run_once(&(pos + I64Vec3::new(x as i64, y as i64, z as i64)))
                }
            }
        }
        res
    }
}

pub trait FunctionOpType {
    fn name(&self) -> ResourceLocation<DynFnOpType>;
    fn create_from_json(
        &self,
        json: &Map<String, Value>,
        reg: &mut Registries,
    ) -> Result<Box<dyn FunctionOp>, Arc<Error>>;
}

pub type DynFnOpType = Box<dyn FunctionOpType + Send + Sync>;

/// Creates a new [`Error::JsonError`] from a string, wrapped in an [`Arc`].
fn json_err(text: impl Into<String>) -> Arc<Error> {
    Arc::new(Error::JsonParseError(text.into()))
}

/// Creates a new [`FunctionOp`] from some JSON.
pub fn from_json(json: &Value, reg: &mut Registries) -> Result<Box<dyn FunctionOp>, Arc<Error>> {
    match json {
        Value::Number(num) => Ok(Box::new(Constant(num.as_f64().expect("should never fail")))),
        Value::String(s) => {
            let df = reg.get_or_load::<DensityFunction>(
                &ResourceLocation::from_str(s).expect("infallible"),
            )?;
            from_json(&df.0, reg)
        }
        Value::Object(obj) => {
            let ty = obj
                .get("type")
                .ok_or_else(|| json_err("expected `type` field in density function"))?
                .as_str()
                .ok_or_else(|| {
                    json_err("expected `type` field in density function to be a resource location")
                })?
                .parse()
                .expect("infallible");
            let fn_op = reg
                .get::<DynFnOpType>(&ty)
                .ok_or_else(|| Error::UnknownDensityFunctionOp(ty))?;
            fn_op.create_from_json(obj, reg)
        }
        _ => Err(json_err(
            "expected number, resource location or density function",
        )),
    }
}

/// A helper for registering a [`FunctionOpType`] on a [`Registries`].
pub fn register_op(reg: &mut Registries, op: impl FunctionOpType + Send + Sync + 'static) {
    reg.insert(op.name(), Box::new(op));
}

/// A trait for decoding [`FunctionOp`]s from a JSON object.
pub trait FromJson: FunctionOp + Sized + 'static {
    const RES_LOC: ResourceLocation<DynFnOpType>;
    type Json: for<'de> Deserialize<'de>;
    fn from_json(json: Self::Json, reg: &mut Registries) -> Result<Self, Arc<Error>>;
}

/// A [`FunctionOpType`] for [`FunctionOp`]s that implement [`FromJson`].
pub struct JsonOp<T: FromJson>(PhantomData<T>);

impl<T: FromJson> JsonOp<T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: FromJson> Default for JsonOp<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FromJson> FunctionOpType for JsonOp<T> {
    fn name(&self) -> ResourceLocation<DynFnOpType> {
        T::RES_LOC
    }

    fn create_from_json(
        &self,
        json: &Map<String, Value>,
        reg: &mut Registries,
    ) -> Result<Box<dyn FunctionOp>, Arc<Error>> {
        let json: T::Json =
            serde_json::from_value(Value::Object(json.clone())).map_err(|e| Arc::new(e.into()))?;
        T::from_json(json, reg).map(|v| Box::new(v) as Box<dyn FunctionOp>)
    }
}

/// A [`FunctionOpType`] for a [`FunctionOp`] with no arguments.
pub struct UnitOp<F, T>
where
    F: Fn() -> T,
    T: FunctionOp + 'static,
{
    constructor: F,
    name: ResourceLocation<DynFnOpType>,
    _marker: PhantomData<T>,
}

impl<F, T> UnitOp<F, T>
where
    F: Fn() -> T,
    T: FunctionOp + 'static,
{
    pub fn new(constructor: F, name: ResourceLocation<DynFnOpType>) -> Self {
        Self {
            constructor,
            name,
            _marker: PhantomData,
        }
    }

    /// Creates a new [`MonadicOp`] within the `minecraft` namespace
    pub fn new_mc(constructor: F, name: &'static str) -> Self {
        Self::new(constructor, ResourceLocation::new_static_mc(name))
    }
}

impl<F, T> FunctionOpType for UnitOp<F, T>
where
    F: Fn() -> T,
    T: FunctionOp + 'static,
{
    fn name(&self) -> mcpackloader::ResourceLocation<DynFnOpType> {
        self.name.clone()
    }

    fn create_from_json(
        &self,
        _json: &Map<String, Value>,
        _registry: &mut Registries,
    ) -> Result<Box<dyn FunctionOp>, Arc<Error>> {
        Ok(Box::new((self.constructor)()))
    }
}

/// A [`FunctionOpType`] for a [`FunctionOp`] with one argument.
pub struct MonadicOp<F, T>
where
    F: Fn(Box<dyn FunctionOp>) -> T,
    T: FunctionOp + 'static,
{
    constructor: F,
    name: ResourceLocation<DynFnOpType>,
    _marker: PhantomData<T>,
}

impl<F, T> MonadicOp<F, T>
where
    F: Fn(Box<dyn FunctionOp>) -> T,
    T: FunctionOp + 'static,
{
    pub fn new(constructor: F, name: ResourceLocation<DynFnOpType>) -> Self {
        Self {
            constructor,
            name,
            _marker: PhantomData,
        }
    }

    /// Creates a new [`MonadicOp`] within the `minecraft` namespace
    pub fn new_mc(constructor: F, name: &'static str) -> Self {
        Self::new(constructor, ResourceLocation::new_static_mc(name))
    }
}

impl<F, T> FunctionOpType for MonadicOp<F, T>
where
    F: Fn(Box<dyn FunctionOp>) -> T,
    T: FunctionOp + 'static,
{
    fn name(&self) -> mcpackloader::ResourceLocation<DynFnOpType> {
        self.name.clone()
    }

    fn create_from_json(
        &self,
        json: &Map<String, Value>,
        registry: &mut Registries,
    ) -> Result<Box<dyn FunctionOp>, Arc<Error>> {
        let arg_json = json
            .get("argument")
            .ok_or_else(|| json_err("expected `argument` field"))?;
        let arg = from_json(arg_json, registry)?;
        Ok(Box::new((self.constructor)(arg)))
    }
}

/// A [`FunctionOpType`] for a [`FunctionOp`] with two arguments.
pub struct DiadicOp<F, T>
where
    F: Fn(Box<dyn FunctionOp>, Box<dyn FunctionOp>) -> T,
    T: FunctionOp + 'static,
{
    constructor: F,
    name: ResourceLocation<DynFnOpType>,
    _marker: PhantomData<T>,
}

impl<F, T> DiadicOp<F, T>
where
    F: Fn(Box<dyn FunctionOp>, Box<dyn FunctionOp>) -> T,
    T: FunctionOp + 'static,
{
    pub fn new(constructor: F, name: ResourceLocation<DynFnOpType>) -> Self {
        Self {
            constructor,
            name,
            _marker: PhantomData,
        }
    }

    /// Creates a new [`DiadicOp`] within the `minecraft` namespace
    pub fn new_mc(constructor: F, name: &'static str) -> Self {
        Self::new(constructor, ResourceLocation::new_static_mc(name))
    }
}

impl<F, T> FunctionOpType for DiadicOp<F, T>
where
    F: Fn(Box<dyn FunctionOp>, Box<dyn FunctionOp>) -> T,
    T: FunctionOp + 'static,
{
    fn name(&self) -> mcpackloader::ResourceLocation<DynFnOpType> {
        self.name.clone()
    }

    fn create_from_json(
        &self,
        json: &Map<String, Value>,
        registry: &mut Registries,
    ) -> Result<Box<dyn FunctionOp>, Arc<Error>> {
        let arg1_json = json
            .get("argument1")
            .ok_or_else(|| json_err("expected `argument`"))?;
        let arg1 = from_json(arg1_json, registry)?;
        let arg2_json = json
            .get("argument2")
            .ok_or_else(|| json_err("expected `argument`"))?;
        let arg2 = from_json(arg2_json, registry)?;
        Ok(Box::new((self.constructor)(arg1, arg2)))
    }
}
