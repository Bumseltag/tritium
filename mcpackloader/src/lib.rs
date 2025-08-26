use std::{
    any::type_name,
    borrow::Cow,
    error::Error,
    fmt::{Debug, Display},
    hash::Hash,
    io,
    marker::PhantomData,
    path::{Path, PathBuf},
    str::FromStr,
    time::Duration,
};

use image::ImageError;
use serde::{Deserialize, de::Visitor};
use serde_json::Value;
use thiserror::Error;

pub mod blockstates;
pub mod models;
pub mod textures;
pub mod worldgen;

pub struct ResourceLocation<T>(UntypedResourceLocation, PhantomData<T>);

impl<T> ResourceLocation<T> {
    pub const fn air() -> Self {
        Self::new_static_mc("air")
    }

    pub const fn new_static_mc(path: &'static str) -> Self {
        Self(UntypedResourceLocation::new_static_mc(path), PhantomData)
    }

    /// Creates a new [`ResourceLocation`] from a `namespace` and `path`
    pub fn new(namespace: impl Into<String>, path: impl Into<String>) -> Self {
        Self(UntypedResourceLocation::new(namespace, path), PhantomData)
    }

    /// Creates a new [`ResourceLocation`] from a `path` with the namespace `minecraft`
    pub fn new_mc(path: impl Into<String>) -> Self {
        Self(UntypedResourceLocation::new_mc(path), PhantomData)
    }

    pub fn namespace(&self) -> &str {
        self.0.namespace()
    }

    pub fn path(&self) -> &str {
        self.0.path()
    }

    pub fn prefix(&mut self, path: &str) {
        self.0.prefix(path);
    }

    pub fn with_prefix(self, path: &str) -> Self {
        ResourceLocation(self.0.with_prefix(path), PhantomData)
    }

    pub fn suffix(&mut self, path: &str) {
        self.0.suffix(path);
    }

    pub fn with_suffix(self, path: &str) -> Self {
        ResourceLocation(self.0.with_suffix(path), PhantomData)
    }

    pub fn to_assets_path(&self, directory: impl AsRef<Path>, ext: &str) -> PathBuf {
        let mut path = PathBuf::from_str("assets").expect("wtf");
        path.push(self.namespace());
        path.push(directory);
        path.push(self.path());
        path.set_extension(ext);
        path
    }

    pub fn to_data_path(&self, directory: impl AsRef<Path>, ext: &str) -> PathBuf {
        let mut path = PathBuf::from_str("data").expect("wtf");
        path.push(self.namespace());
        path.push(directory);
        path.push(self.path());
        path.set_extension(ext);
        path
    }
}

impl<T> Clone for ResourceLocation<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<T> Debug for ResourceLocation<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ResourceLocation<{}>{{{}}}", type_name::<T>(), self.0)
    }
}

impl<T> Display for ResourceLocation<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T> PartialEq for ResourceLocation<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for ResourceLocation<T> {}

impl<T> Hash for ResourceLocation<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
    }
}

impl<'de, T> Deserialize<'de> for ResourceLocation<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self(
            UntypedResourceLocation::deserialize(deserializer)?,
            PhantomData,
        ))
    }
}

impl<'a, T> From<&'a ResourceLocation<T>> for ResourceLocation<T> {
    fn from(value: &'a ResourceLocation<T>) -> Self {
        value.clone()
    }
}

impl<T> From<UntypedResourceLocation> for ResourceLocation<T> {
    fn from(value: UntypedResourceLocation) -> Self {
        ResourceLocation(value, PhantomData)
    }
}

impl<T> FromStr for ResourceLocation<T> {
    type Err = Never;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(UntypedResourceLocation::from_str(s)?.into())
    }
}

// the compiler automatically adds a trait bound of `T: Send` and `T: Sync`,
// but we want to override those.
unsafe impl<T> Send for ResourceLocation<T> {}
unsafe impl<T> Sync for ResourceLocation<T> {}

// #[derive(Debug, Clone, PartialEq, Eq)]
// enum MaybeOwnedStr {
//     Owned(String),
//     Static(&'static str),
// }

// impl Hash for MaybeOwnedStr {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         match self {
//             MaybeOwnedStr::Owned(own) => {
//                 own.hash(state);
//             }
//             MaybeOwnedStr::Static(st) => {
//                 st.hash(state);
//             }
//         }
//     }
// }

// impl Borrow<str> for MaybeOwnedStr {
//     fn borrow(&self) -> &str {
//         match self {
//             MaybeOwnedStr::Owned(own) => &own,
//             MaybeOwnedStr::Static(st) => st,
//         }
//     }
// }

/// An untyped resource location
///
/// A resource location consists of a namespace and a path,
/// seperated by a `:`, and usually only consist of
/// lowercase ascii characters and underscores.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct UntypedResourceLocation {
    namespace: Cow<'static, str>,
    path: Cow<'static, str>,
}

impl UntypedResourceLocation {
    /// Creates a new [`UntypedResourceLocation`] with value `"minecraft:air"`
    pub const fn air() -> Self {
        Self::new_static_mc("air")
    }

    /// Creates a new [`UntypedResourceLocation`] from a `path` with the namespace `minecraft`.
    ///
    /// Note that this function is `const`
    pub const fn new_static_mc(path: &'static str) -> Self {
        Self::new_static("minecraft", path)
    }

    pub const fn new_static(namespace: &'static str, path: &'static str) -> Self {
        Self {
            namespace: Cow::Borrowed(namespace),
            path: Cow::Borrowed(path),
        }
    }

    /// Creates a new [`UntypedResourceLocation`] from a `namespace` and `path`
    pub fn new(namespace: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            namespace: Cow::Owned(namespace.into()),
            path: Cow::Owned(path.into()),
        }
    }

    /// Creates a new [`UntypedResourceLocation`] from a `path` with the namespace `minecraft`
    pub fn new_mc(path: impl Into<String>) -> Self {
        Self {
            namespace: Cow::Borrowed("minecraft"),
            path: Cow::Owned(path.into()),
        }
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn prefix(&mut self, path: &str) {
        self.path.to_mut().insert_str(0, path);
    }

    pub fn with_prefix(mut self, path: &str) -> Self {
        self.prefix(path);
        self
    }

    pub fn suffix(&mut self, path: &str) {
        self.path.to_mut().push_str(path);
    }

    pub fn with_suffix(mut self, path: &str) -> Self {
        self.suffix(path);
        self
    }

    pub fn to_assets_path(&self, directory: impl AsRef<Path>) -> PathBuf {
        let mut path = PathBuf::from_str("assets").expect("wtf");
        path.push(self.namespace.as_ref());
        path.push(directory);
        path.push(self.path.as_ref());
        path
    }
}

impl FromStr for UntypedResourceLocation {
    type Err = Never;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let split_path = s.split_once(':');
        if let Some((ns, path)) = split_path {
            Ok(UntypedResourceLocation::new(ns, path))
        } else {
            Ok(UntypedResourceLocation::new_mc(s))
        }
    }
}

impl<'de> Deserialize<'de> for UntypedResourceLocation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(ResourceLocationVisitor)
    }
}

#[derive(Debug)]
struct ResourceLocationVisitor;

impl<'de> Visitor<'de> for ResourceLocationVisitor {
    type Value = UntypedResourceLocation;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a resource location")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(UntypedResourceLocation::from_str(v).unwrap())
    }
}

#[derive(Debug, Error)]
pub enum Never {}

impl Display for UntypedResourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.namespace, self.path)
    }
}

pub trait ResourceType: Sized {
    fn to_path(res_loc: &ResourceLocation<Self>) -> PathBuf;

    /// Opens and reads the resource.
    ///
    /// If the `async` feature is enabled, you should consider using [`ResourceType::open_async`] instead.
    fn open(path: PathBuf) -> Result<Self, ResourceParseError>;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ticks(u64);

impl Ticks {
    pub fn new(ticks: u64) -> Self {
        Self(ticks)
    }

    pub fn ticks(self) -> u64 {
        self.0
    }

    pub fn to_duration(self) -> Duration {
        Duration::from_millis(self.0 * (1000 / 20))
    }

    pub fn from_duration(duration: Duration) -> Self {
        Self::new(duration.as_millis() as u64 / (1000 / 20))
    }
}

#[derive(Debug, Error)]
pub enum ResourceParseError {
    #[error(
        "Invalid type in JSON, for correct schema see https://minecraft.wiki/w/Tutorial:Models#Block_states \nFound: {0:#?}"
    )]
    InvalidType(Value),
    #[error(
        "Invalid blockstates provided for variant, expected comma-seperated list of `<prop>=<value>` statements, found: {0}"
    )]
    InvalidVariantBlockstates(String),
    #[error("Invalid format: {0}")]
    InvalidFormat(&'static str),
    #[error(
        "Empty `when` clause of a multipart was found. Expected a field with name `AND`, `OR`, or a blockstate."
    )]
    EmptyMultipartWhen,
    #[error("Multipart part is missing `apply` clause.")]
    NoApplyClause,
    #[error("Resource not found. Checked the following locations: {0:?}.")]
    ResourceNotFound(Vec<String>),
    #[error("Error occurred while parsing JSON")]
    Json(#[from] serde_json::Error),
    #[error("IO error occurred")]
    Io(#[from] io::Error),
    #[error("Image parse error occurred")]
    Image(#[from] ImageError),
    #[error("{0}")]
    String(String),
}

impl From<&ResourceParseError> for ResourceParseError {
    fn from(value: &ResourceParseError) -> Self {
        ResourceParseError::String(
            value.to_string() + &format!("\n Caused by: {:?}", value.source()),
        )
    }
}

impl From<&str> for ResourceParseError {
    fn from(value: &str) -> Self {
        ResourceParseError::String(value.to_string())
    }
}

impl From<String> for ResourceParseError {
    fn from(value: String) -> Self {
        ResourceParseError::String(value)
    }
}
