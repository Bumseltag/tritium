use std::{
    ops::{Add, Range, RangeInclusive},
    sync::Mutex,
};

use glam::IVec3;

use crate::helpers::{
    get_seed, java_str_hash, transmute_to_i32, transmute_to_i64, transmute_to_u64,
    u128_seed_from_hash_of,
};

/// A random number generator that can be shared across threads.
///
/// Equivalent to `net.minecraft.util.RandomSource`.
pub trait Rng: Sync {
    type Fork: Rng;
    type ForkFactory: RngFactory;

    /// Sets the seed of the rng
    fn set_seed(&self, seed: i64);

    /// Generates a new random `i32` between `i32::MIN..=i32::MAX`
    fn next_i32(&self) -> i32;

    /// Generates a new unsigned `i32` between `0..bound` (excluding `bound`).
    ///
    /// Panics if `bound` is negative.
    fn next_i32_up_to(&self, bound: i32) -> i32;

    /// Generates a new `i32` within the given bound.
    fn next_i32_between(&self, bounds: impl IntoRange<i32>) -> i32 {
        let bounds = bounds.into_range();
        self.next_i32_up_to(bounds.end - bounds.start) + bounds.start
    }

    /// Generates a new random `i64` between `i64::MIN..=i64::MAX`.
    fn next_i64(&self) -> i64;

    /// Generates a new random `bool`.
    fn next_bool(&self) -> bool;

    /// Generates a new `f32` between `0.0..=1.0`.
    fn next_f32(&self) -> f32;

    /// Generates a new `f64` between `0.0..=1.0`.
    fn next_f64(&self) -> f64;

    /// Generates a new `f64` according to the normal distribution.
    fn next_gaussian(&self) -> f64;

    /// Generates a new `f64` according to a triangle distribution,
    /// where `base` is most likely and `base - amplitude` and `base + amplitude` are least likely.
    fn triangle(&self, base: f64, amplitude: f64) -> f64 {
        base + (amplitude * (self.next_f64() - self.next_f64()))
    }

    /// Generates and discards `n` `i32`s, effectively advancing the seed.
    fn skip_i32(&self, count: usize) {
        for _ in 0..count {
            self.next_i32();
        }
    }

    /// Clones this RNG.
    fn fork(&self) -> Self::Fork;

    /// Creates an [`RngFactory`].
    fn fork_factory(&self) -> Self::ForkFactory;
}

/// Minecrafts main RNG method (no idea why its called "legacy").
///
/// equivalent to `net.minecraft.world.level.levelgen.LegacyRandomSource`.
#[derive(Debug)]
pub struct LegacyRng {
    seed: Mutex<i64>,
    gaussian_rng: Mutex<MarsagliaPolarGaussian>,
}

impl LegacyRng {
    pub fn new(seed: i64) -> Self {
        Self {
            seed: Mutex::new((seed ^ 25214903917) & 281474976710655),
            gaussian_rng: Mutex::new(MarsagliaPolarGaussian::new()),
        }
    }

    fn next(&self, n: u32) -> i32 {
        let mut seed = self.seed.lock().unwrap();
        let random_val = seed.wrapping_mul(25214903917).wrapping_add(11) & 281474976710655;
        *seed = random_val;
        drop(seed);

        (random_val >> (48 - n)) as i32
    }
}

impl Clone for LegacyRng {
    fn clone(&self) -> Self {
        Self::new(*self.seed.lock().unwrap())
    }
}

impl Rng for LegacyRng {
    type Fork = LegacyRng;
    type ForkFactory = LegacyRngFactory;

    fn set_seed(&self, seed: i64) {
        *self.seed.lock().unwrap() = (seed ^ 25214903917) & 281474976710655;
    }

    fn next_i32(&self) -> i32 {
        self.next(32)
    }

    fn next_i32_up_to(&self, bound: i32) -> i32 {
        debug_assert!(bound > 0, "bound must be positive");

        // probably hurts performance more than it helps lol
        if bound.count_ones() == 1 {
            return ((self.next(31) as i64 * bound as i64) >> 31) as i32;
        }

        // the java code does the following, however the loop seems completely unnecessary...
        // let mut a = self.next(31);
        // let mut b = a % bound;
        // while (a - b) + (bound - 1) < 0 {
        //     a = self.next(31);
        //     b = a & bound;
        // }
        // return b;

        self.next(31) % bound
    }

    fn next_i64(&self) -> i64 {
        ((self.next(32) as i64) << 32) + self.next(32) as i64
    }

    fn next_bool(&self) -> bool {
        self.next(1) != 0
    }

    fn next_f32(&self) -> f32 {
        self.next(24) as f32 * F32_MULTIPLIER
    }

    fn next_f64(&self) -> f64 {
        let long = ((self.next(26) as i64) << 27) | self.next(27) as i64;
        long as f64 * F64_MULTIPLIER
    }

    fn next_gaussian(&self) -> f64 {
        self.gaussian_rng.lock().unwrap().next_gaussian(self)
    }

    fn fork(&self) -> Self::Fork {
        self.clone()
    }

    fn fork_factory(&self) -> Self::ForkFactory {
        LegacyRngFactory::new(self.next_i64())
    }
}

const F32_MULTIPLIER: f32 = 1.0 / ((1i64 << 24) as f32);
const F64_MULTIPLIER: f64 = 1.0 / ((1i64 << 53) as f64);

impl Default for LegacyRng {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Implements the Marsaglia polar method for generating sampling a number from the normal distribution.
///
/// Equivalent to `net.minecraft.world.level.levelgen.MarsagliaPolarGaussian`
#[derive(Debug, Clone, Default)]
pub struct MarsagliaPolarGaussian {
    next_next_gaussian: Option<f64>,
}

impl MarsagliaPolarGaussian {
    /// Creates a new [`MarsagliaPolarGaussian`]
    pub fn new() -> Self {
        Self::default()
    }

    /// Resets the rng.
    pub fn reset(&mut self) {
        self.next_next_gaussian = None;
    }

    /// Generates a new number sampled from the normal distribution.
    pub fn next_gaussian<T: Rng>(&mut self, rng: &T) -> f64 {
        if let Some(result) = self.next_next_gaussian.take() {
            return result;
        }

        let mut a = 2.0 * rng.next_f64() - 1.0;
        let mut b = 2.0 * rng.next_f64() - 1.0;
        let mut c = (a * a) + (b * b);
        while c >= 1.0 || c == 0.0 {
            a = 2.0 * rng.next_f64() - 1.0;
            b = 2.0 * rng.next_f64() - 1.0;
            c = (a * a) + (b * b);
        }
        let d = (-2.0 * c.ln() / c).sqrt();
        self.next_next_gaussian = Some(b * d);
        a * d
    }
}

/// The rng used for worldgen, based on [`LegacyRng`].
///
/// Equivalent of `net.minecraft.world.level.levelgen.LegacyRandomSource`
pub struct WorldgenRng {
    main: LegacyRng,
    base: LegacyRng,
}

impl WorldgenRng {
    pub fn new(rng: LegacyRng) -> Self {
        Self {
            main: LegacyRng::new(0),
            base: rng,
        }
    }
}

impl Rng for WorldgenRng {
    type Fork = LegacyRng;
    type ForkFactory = LegacyRngFactory;

    fn fork(&self) -> Self::Fork {
        self.base.fork()
    }

    fn fork_factory(&self) -> Self::ForkFactory {
        self.base.fork_factory()
    }

    fn set_seed(&self, seed: i64) {
        self.main.set_seed(seed);
    }
    fn next_i32(&self) -> i32 {
        self.main.next_i32()
    }
    fn next_i32_up_to(&self, bound: i32) -> i32 {
        self.main.next_i32_up_to(bound)
    }
    fn next_i32_between(&self, bounds: impl IntoRange<i32>) -> i32 {
        self.main.next_i32_between(bounds)
    }
    fn next_i64(&self) -> i64 {
        self.main.next_i64()
    }
    fn next_f32(&self) -> f32 {
        self.main.next_f32()
    }
    fn next_f64(&self) -> f64 {
        self.main.next_f64()
    }
    fn next_bool(&self) -> bool {
        self.main.next_bool()
    }
    fn next_gaussian(&self) -> f64 {
        self.main.next_gaussian()
    }
    fn triangle(&self, base: f64, amplitude: f64) -> f64 {
        self.main.triangle(base, amplitude)
    }
    fn skip_i32(&self, count: usize) {
        self.main.skip_i32(count);
    }
}

pub trait RngFactory {
    type Rng: Rng;
    fn at(&self, pos: IVec3) -> Self::Rng;
    fn hash_of(&self, text: &str) -> Self::Rng;
}

/// The [`RngFactory`] of [`LegacyRng`]
pub struct LegacyRngFactory {
    seed: i64,
}

impl LegacyRngFactory {
    pub fn new(seed: i64) -> Self {
        Self { seed }
    }
}

impl RngFactory for LegacyRngFactory {
    type Rng = LegacyRng;

    fn at(&self, pos: IVec3) -> Self::Rng {
        let pos_seed = get_seed(pos);
        LegacyRng::new(pos_seed ^ self.seed)
    }

    fn hash_of(&self, text: &str) -> Self::Rng {
        let hash = java_str_hash(text);
        LegacyRng::new(hash as i64 ^ self.seed)
    }
}

/// A rng based on [`Xoroshiro128PlusPlus`].
///
/// Equivalent to `net.minecraft.world.level.levelgen.XoroshiroRandomSource`
pub struct XoroshiroRng {
    inner: Mutex<Xoroshiro128PlusPlus>,
    gaussian_rng: Mutex<MarsagliaPolarGaussian>,
}

impl XoroshiroRng {
    pub fn from_u128_seed(seed_lo: u64, seed_hi: u64) -> Self {
        Self {
            inner: Mutex::new(Xoroshiro128PlusPlus::new(seed_lo, seed_hi)),
            gaussian_rng: Mutex::new(MarsagliaPolarGaussian::new()),
        }
    }

    pub fn new(seed: i64) -> Self {
        let (seed_lo, seed_hi) = Xoroshiro128PlusPlus::u128_seed_from_u64(transmute_to_u64(seed));
        Self::from_u128_seed(seed_lo, seed_hi)
    }

    pub fn next_u32(&self) -> u32 {
        (self.next_u64() & 0xFFFFFFFF) as u32
    }

    pub fn next_u64(&self) -> u64 {
        self.inner.lock().unwrap().next_u64()
    }

    pub fn next_bits(&self, i: u64) -> u64 {
        self.next_u64() >> (64 - i)
    }
}

impl Rng for XoroshiroRng {
    type Fork = XoroshiroRng;
    type ForkFactory = XoroshiroRngFacory;

    fn set_seed(&self, seed: i64) {
        let new_seed = Xoroshiro128PlusPlus::u128_seed_from_u64(transmute_to_u64(seed));
        *self.inner.lock().unwrap() = Xoroshiro128PlusPlus::new(new_seed.0, new_seed.1);
    }

    fn next_i32(&self) -> i32 {
        transmute_to_i32(self.next_u32())
    }

    fn next_i32_up_to(&self, bound: i32) -> i32 {
        debug_assert!(bound > 0, "bound must be positive");
        let bound = bound as u32;
        let mut l = self.next_u32() as u64;
        let mut m = l.wrapping_mul(bound as u64);
        let mut n = m & 4294967295;
        if n < bound as u64 {
            let i = (!bound + 1) % bound;
            while n < transmute_to_u64(transmute_to_i32(i) as i64) {
                l = self.next_u32() as u64;
                m = l.wrapping_mul(bound as u64);
                n = m & 4294967295;
            }
        }

        (transmute_to_i64(m) >> 32) as i32
    }

    fn next_i64(&self) -> i64 {
        transmute_to_i64(self.inner.lock().unwrap().next_u64())
    }

    fn next_bool(&self) -> bool {
        self.next_i64() & 1 != 0
    }

    fn next_f32(&self) -> f32 {
        self.next_bits(24) as f32 * F32_MULTIPLIER
    }

    fn next_f64(&self) -> f64 {
        self.next_bits(53) as f64 * F64_MULTIPLIER
    }

    fn next_gaussian(&self) -> f64 {
        self.gaussian_rng.lock().unwrap().next_gaussian(self)
    }

    fn fork(&self) -> Self::Fork {
        Self::from_u128_seed(self.next_u64(), self.next_u64())
    }

    fn fork_factory(&self) -> Self::ForkFactory {
        XoroshiroRngFacory::new(self.next_u64(), self.next_u64())
    }
}

impl Clone for XoroshiroRng {
    fn clone(&self) -> Self {
        Self {
            inner: Mutex::new(self.inner.lock().unwrap().clone()),
            gaussian_rng: Mutex::new(MarsagliaPolarGaussian::new()),
        }
    }
}

impl Default for XoroshiroRng {
    fn default() -> Self {
        Self::new(0)
    }
}

/// The [`RngFactory`] of [`XoroshiroRng`].
pub struct XoroshiroRngFacory {
    seed_lo: u64,
    seed_hi: u64,
}

impl XoroshiroRngFacory {
    pub fn new(seed_lo: u64, seed_hi: u64) -> Self {
        Self { seed_lo, seed_hi }
    }
}

impl RngFactory for XoroshiroRngFacory {
    type Rng = XoroshiroRng;

    fn at(&self, pos: IVec3) -> Self::Rng {
        let pos_seed = transmute_to_u64(get_seed(pos));
        XoroshiroRng::from_u128_seed(pos_seed ^ self.seed_lo, self.seed_hi)
    }

    fn hash_of(&self, text: &str) -> Self::Rng {
        let (seed_lo, seed_hi) = u128_seed_from_hash_of(text);
        XoroshiroRng::from_u128_seed(seed_lo ^ self.seed_lo, seed_hi ^ self.seed_hi)
    }
}

/// The Xoroshiro128++ PRNG, See <https://prng.di.unimi.it/xoroshiro128plusplus.c>
#[derive(Debug, Clone)]
pub struct Xoroshiro128PlusPlus {
    seed_lo: u64,
    seed_hi: u64,
}

impl Xoroshiro128PlusPlus {
    pub fn new(seed_lo: u64, seed_hi: u64) -> Self {
        let (seed_lo, seed_hi) = if seed_lo | seed_hi == 0 {
            (
                transmute_to_u64(Self::GOLDEN_RATIO_64),
                transmute_to_u64(Self::SILVER_RATIO_64),
            )
        } else {
            (seed_lo, seed_hi)
        };
        Self { seed_lo, seed_hi }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = (self.seed_lo.wrapping_add(self.seed_hi))
            .rotate_left(17)
            .wrapping_add(self.seed_lo);
        let new_seed = self.seed_hi ^ self.seed_lo;
        self.seed_lo = self.seed_lo.rotate_left(49) ^ new_seed ^ (new_seed << 21);
        self.seed_hi = new_seed.rotate_left(28);
        result
    }

    pub fn mix_stafford13(mut value: u64) -> u64 {
        value = (value ^ (value >> 30)).wrapping_mul(transmute_to_u64(-4658895280553007687));
        value = (value ^ (value >> 27)).wrapping_mul(transmute_to_u64(-7723592293110705685));
        value ^ (value >> 31)
    }

    pub fn u128_seed_from_u64(seed: u64) -> (u64, u64) {
        let lo = seed ^ transmute_to_u64(Self::SILVER_RATIO_64);
        let hi = lo.wrapping_add(transmute_to_u64(Self::GOLDEN_RATIO_64));
        (Self::mix_stafford13(lo), Self::mix_stafford13(hi))
    }

    const GOLDEN_RATIO_64: i64 = -7046029254386353131;
    const SILVER_RATIO_64: i64 = 7640891576956012809;
}

pub trait IntoRange<T> {
    fn into_range(self) -> Range<T>;
}

impl<T> IntoRange<T> for Range<T> {
    fn into_range(self) -> Range<T> {
        self
    }
}

impl<T: Copy + Add<Output = T> + From<u8>> IntoRange<T> for RangeInclusive<T> {
    fn into_range(self) -> Range<T> {
        (*self.start())..(*self.end() + T::from(1))
    }
}

#[cfg(all(test, feature = "java_tests_module"))]
mod java_tests {
    use jni::objects::{JObject, JValue};

    use crate::{
        helpers::{transmute_to_i64, transmute_to_u64},
        java_tests::{Class, Env, RandomSource, RandomSupport, get_jvm_env},
        random::{LegacyRng, Rng, Xoroshiro128PlusPlus, XoroshiroRng},
    };

    #[test]
    fn legacy_rng() {
        let mut env = get_jvm_env();
        println!("testing seed 0");
        let random_source = env.construct(
            &Class::LegacyRandomSource,
            RandomSource::CONSTRUCTOR,
            &[JValue::Long(0)],
        );
        test_rng(&mut env, &LegacyRng::new(0), &random_source);

        println!("testing seed 123");
        let random_source = env.construct(
            &Class::LegacyRandomSource,
            RandomSource::CONSTRUCTOR,
            &[JValue::Long(123)],
        );
        test_rng(&mut env, &LegacyRng::new(123), &random_source);
    }

    #[test]
    fn xoroshiro_rng() {
        let mut env = get_jvm_env();
        println!("testing seed 0");
        let random_source = env.construct(
            &Class::XoroshiroRandomSource,
            RandomSource::CONSTRUCTOR,
            &[JValue::Long(0)],
        );
        test_rng(&mut env, &XoroshiroRng::new(0), &random_source);

        println!("testing seed 123");
        let random_source = env.construct(
            &Class::XoroshiroRandomSource,
            RandomSource::CONSTRUCTOR,
            &[JValue::Long(123)],
        );
        test_rng(&mut env, &XoroshiroRng::new(123), &random_source);
    }

    #[test]
    fn mix_stafford13() {
        let mut env = get_jvm_env();
        let nums = [0, 5345, 34569031, 90243562349, -153453, -1452349523452];
        for num in nums {
            assert_eq!(
                env.call_static(&RandomSupport::MIX_STAFFORD13, &[JValue::Long(num)])
                    .j()
                    .unwrap(),
                transmute_to_i64(Xoroshiro128PlusPlus::mix_stafford13(transmute_to_u64(num))),
                "input: {num}"
            );
        }
    }

    macro_rules! assert_eq_10x {
        ($env:expr, $java_rng:ident.$java_fn:ident($($java_args:expr),*) -> $out:ident, $rust_expr:expr) => {
            for i in 0..10 {
                let java_res = $env.call($java_rng, &RandomSource::$java_fn, &[$($java_args),*]).$out().unwrap();
                let rust_res = $rust_expr;
                assert_eq!(java_res, rust_res, "iteration {i}");
            }
        };
    }

    fn test_rng<T: Rng>(env: &mut Env, rng: &T, java_rng: &JObject) {
        assert_eq_10x!(env, java_rng.NEXT_INT() -> i, rng.next_i32());
        assert_eq_10x!(env, java_rng.NEXT_INT_UP_TO(JValue::Int(10)) -> i, rng.next_i32_up_to(10));
        assert_eq_10x!(env, java_rng.NEXT_INT_UP_TO(JValue::Int(8)) -> i, rng.next_i32_up_to(8));
        assert_eq_10x!(env, java_rng.NEXT_INT_BEWTEEN(JValue::Int(3), JValue::Int(6)) -> i, rng.next_i32_between(3..6));
        assert_eq_10x!(env, java_rng.NEXT_LONG() -> j, rng.next_i64());
        assert_eq_10x!(env, java_rng.NEXT_FLOAT() -> f, rng.next_f32());
        assert_eq_10x!(env, java_rng.NEXT_DOUBLE() -> d, rng.next_f64());
        assert_eq_10x!(env, java_rng.NEXT_BOOLEAN() -> z, rng.next_bool());
        assert_eq_10x!(env, java_rng.NEXT_GAUSSIAN() -> d, rng.next_gaussian());
        assert_eq_10x!(env, java_rng.TRIANGLE(JValue::Double(2.0), JValue::Double(4.0)) -> d, rng.triangle(2.0, 4.0));
    }
}
