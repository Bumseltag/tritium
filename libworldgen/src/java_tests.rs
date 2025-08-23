use std::sync::LazyLock;

use jni::{
    AttachGuard, InitArgsBuilder, JavaVM,
    objects::{JClass, JObject, JString, JValue, JValueOwned},
};

static JVM: LazyLock<JavaVM> = LazyLock::new(create_jvm);

fn create_jvm() -> JavaVM {
    let args = InitArgsBuilder::new()
        .version(jni::JNIVersion::V8)
        .option("-Djava.class.path=libworldgen-java-tests/app/build/libs/app-uber.jar")
        .build()
        .unwrap();
    JavaVM::new(args).unwrap()
}

pub fn get_jvm_env<'a>() -> Env<'a> {
    Env::new(JVM.attach_current_thread().unwrap())
}

pub struct Classes<'a> {
    legacy_random_source: JClass<'a>,
    xoroshiro_random_source: JClass<'a>,
    random_support: JClass<'a>,
    perlin_noise: JClass<'a>,
    improved_noise: JClass<'a>,
    blended_noise: JClass<'a>,
    simplex_noise: JClass<'a>,
    mth: JClass<'a>,
    double_array_list: JClass<'a>,
}

impl<'a> Classes<'a> {
    fn get_class(&self, class: &Class) -> &JClass<'a> {
        match class {
            Class::LegacyRandomSource => &self.legacy_random_source,
            Class::XoroshiroRandomSource => &self.xoroshiro_random_source,
            Class::RandomSupport => &self.random_support,
            Class::PerlinNoise => &self.perlin_noise,
            Class::ImprovedNoise => &self.improved_noise,
            Class::BlendedNoise => &self.blended_noise,
            Class::SimplexNoise => &self.simplex_noise,
            Class::Mth => &self.mth,
            Class::DoubleArrayList => &self.double_array_list,
        }
    }
}

pub struct Env<'a> {
    attach_guard: AttachGuard<'a>,
    classes: Classes<'a>,
}

impl<'a> Env<'a> {
    fn new(mut attach_guard: AttachGuard<'a>) -> Self {
        Self {
            classes: Classes {
                legacy_random_source: attach_guard
                    .find_class("net/minecraft/world/level/levelgen/LegacyRandomSource")
                    .unwrap(),
                xoroshiro_random_source: attach_guard
                    .find_class("net/minecraft/world/level/levelgen/XoroshiroRandomSource")
                    .unwrap(),
                random_support: attach_guard
                    .find_class("net/minecraft/world/level/levelgen/RandomSupport")
                    .unwrap(),
                perlin_noise: attach_guard
                    .find_class("net/minecraft/world/level/levelgen/synth/PerlinNoise")
                    .unwrap(),
                improved_noise: attach_guard
                    .find_class("net/minecraft/world/level/levelgen/synth/ImprovedNoise")
                    .unwrap(),
                blended_noise: attach_guard
                    .find_class("bumseltag/libworldgen_java_tests/PatchedBlendedNoise")
                    .unwrap(),
                simplex_noise: attach_guard
                    .find_class("net/minecraft/world/level/levelgen/synth/SimplexNoise")
                    .unwrap(),
                mth: attach_guard.find_class("net/minecraft/util/Mth").unwrap(),
                double_array_list: attach_guard
                    .find_class("it/unimi/dsi/fastutil/doubles/DoubleArrayList")
                    .unwrap(),
            },
            attach_guard,
        }
    }

    pub fn construct(&mut self, class: &Class, sig: &str, args: &[JValue]) -> JObject<'a> {
        self.attach_guard
            .new_object(self.classes.get_class(class), sig, args)
            .unwrap()
    }

    pub fn call(&mut self, obj: &JObject, f: &Fn, args: &[JValue]) -> JValueOwned<'a> {
        self.attach_guard
            .call_method(obj, f.name, f.sig, args)
            .unwrap()
    }

    pub fn call_static(&mut self, f: &StaticFn, args: &[JValue]) -> JValueOwned<'a> {
        self.attach_guard
            .call_static_method(self.classes.get_class(&f.class), f.name, f.sig, args)
            .unwrap()
    }

    pub fn string(&mut self, value: &str) -> JString<'a> {
        self.attach_guard.new_string(value).unwrap()
    }

    pub fn field(&mut self, obj: &JObject, field: &Field) -> JValueOwned<'a> {
        self.attach_guard
            .get_field(obj, field.name, field.ty)
            .unwrap()
    }
}

pub enum Class {
    LegacyRandomSource,
    XoroshiroRandomSource,
    RandomSupport,
    ImprovedNoise,
    PerlinNoise,
    BlendedNoise,
    SimplexNoise,
    DoubleArrayList,
    Mth,
}

pub struct Fn {
    pub name: &'static str,
    pub sig: &'static str,
}

pub struct StaticFn {
    pub class: Class,
    pub name: &'static str,
    pub sig: &'static str,
}

pub struct Field {
    pub name: &'static str,
    pub ty: &'static str,
}

pub struct RandomSource;

impl RandomSource {
    pub const CONSTRUCTOR: &'static str = "(J)V";
    pub const NEXT_INT: Fn = Fn {
        name: "nextInt",
        sig: "()I",
    };
    pub const NEXT_INT_UP_TO: Fn = Fn {
        name: "nextInt",
        sig: "(I)I",
    };
    pub const NEXT_INT_BEWTEEN: Fn = Fn {
        name: "nextInt",
        sig: "(II)I",
    };
    pub const NEXT_LONG: Fn = Fn {
        name: "nextLong",
        sig: "()J",
    };
    pub const NEXT_FLOAT: Fn = Fn {
        name: "nextFloat",
        sig: "()F",
    };
    pub const NEXT_DOUBLE: Fn = Fn {
        name: "nextDouble",
        sig: "()D",
    };
    pub const NEXT_BOOLEAN: Fn = Fn {
        name: "nextBoolean",
        sig: "()Z",
    };
    pub const NEXT_GAUSSIAN: Fn = Fn {
        name: "nextGaussian",
        sig: "()D",
    };
    pub const TRIANGLE: Fn = Fn {
        name: "triangle",
        sig: "(DD)D",
    };
}

pub struct RandomSupport;

impl RandomSupport {
    pub const MIX_STAFFORD13: StaticFn = StaticFn {
        class: Class::RandomSupport,
        name: "mixStafford13",
        sig: "(J)J",
    };
    pub const SEED_FROM_HASH_OF: StaticFn = StaticFn {
        class: Class::RandomSupport,
        name: "seedFromHashOf",
        sig: "(Ljava/lang/String;)Lnet/minecraft/world/level/levelgen/RandomSupport$Seed128bit;",
    };
}

pub struct Seed128bit;

impl Seed128bit {
    pub const SEED_LO: Field = Field {
        name: "seedLo",
        ty: "J",
    };
    pub const SEED_HI: Field = Field {
        name: "seedHi",
        ty: "J",
    };
}

pub struct PerlinNoise;

impl PerlinNoise {
    pub const CREATE: StaticFn = StaticFn {
        class: Class::PerlinNoise,
        name: "create",
        sig: "(Lnet/minecraft/util/RandomSource;ILit/unimi/dsi/fastutil/doubles/DoubleList;)Lnet/minecraft/world/level/levelgen/synth/PerlinNoise;",
    };
    pub const GET_VALUE: Fn = Fn {
        name: "getValue",
        sig: "(DDD)D",
    };
    pub const WRAP: StaticFn = StaticFn {
        class: Class::PerlinNoise,
        name: "wrap",
        sig: "(D)D",
    };
}

pub struct ImprovedNoise;

impl ImprovedNoise {
    pub const CONSTRUCTOR: &'static str = "(Lnet/minecraft/util/RandomSource;)V";
    pub const NOISE: Fn = Fn {
        name: "noise",
        sig: "(DDD)D",
    };
    pub const P: Fn = Fn {
        name: "p",
        sig: "(I)I",
    };
    pub const GRAD_DOT: StaticFn = StaticFn {
        class: Class::ImprovedNoise,
        name: "gradDot",
        sig: "(IDDD)D",
    };
    pub const XO: Field = Field {
        name: "xo",
        ty: "D",
    };
    pub const YO: Field = Field {
        name: "yo",
        ty: "D",
    };
    pub const ZO: Field = Field {
        name: "zo",
        ty: "D",
    };
}

pub struct BlendedNoise;

impl BlendedNoise {
    pub const CREATE_UNSEEDED: StaticFn = StaticFn {
        class: Class::BlendedNoise,
        name: "createUnseeded",
        sig: "(DDDDD)Lbumseltag/libworldgen_java_tests/PatchedBlendedNoise;",
    };
    pub const COMPUTE: Fn = Fn {
        name: "compute",
        sig: "(DDD)D",
    };
}

pub struct SimplexNoise;

impl SimplexNoise {
    pub const CONSTRUCTOR: &'static str = "(Lnet/minecraft/util/RandomSource;)V";
    pub const GET_VALUE_2D: Fn = Fn {
        name: "getValue",
        sig: "(DD)D",
    };
}

pub struct Mth;

impl Mth {
    pub const SMOOTHSTEP: StaticFn = StaticFn {
        class: Class::Mth,
        name: "smoothstep",
        sig: "(D)D",
    };

    pub const FLOOR: StaticFn = StaticFn {
        class: Class::Mth,
        name: "floor",
        sig: "(D)I",
    };
}

pub struct DoubleArrayList;

impl DoubleArrayList {
    pub const CONSTRUCTOR: &'static str = "([D)V";

    pub fn create<'a>(env: &mut Env<'a>, values: &[f64]) -> JObject<'a> {
        let arr = env
            .attach_guard
            .new_double_array(values.len() as i32)
            .unwrap();
        env.attach_guard
            .set_double_array_region(&arr, 0, values)
            .unwrap();
        env.construct(
            &Class::DoubleArrayList,
            DoubleArrayList::CONSTRUCTOR,
            &[JValue::Object(&arr)],
        )
    }
}
