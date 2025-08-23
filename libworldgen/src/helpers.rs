use glam::{DVec2, DVec3, IVec3, Vec3Swizzles};
use itertools::Itertools;
use utf16string::{BigEndian, WString};

/// Computes the java `hashCode` function for a string
pub fn java_str_hash(text: &str) -> i32 {
    if !text.is_ascii() {
        return java_str_hash_utf16(text);
    }

    let n = text.len() as u32;
    let mut hash: i32 = 0;
    for (i, b) in text.bytes().enumerate() {
        hash = hash.wrapping_add(
            (b as i32).wrapping_mul(31i32.wrapping_pow(n.wrapping_sub(i as u32 + 1))),
        );
    }
    hash
}

/// Computes the java `hashCode` function for non-ascii strings
fn java_str_hash_utf16(text: &str) -> i32 {
    let utf16_str = WString::<BigEndian>::from(text);
    let mut hash: i32 = 0;
    for b in utf16_str
        .as_bytes()
        .iter()
        .chunks(2)
        .into_iter()
        .map(|mut c| ((*c.next().unwrap() as u16) << 8) | *c.next().unwrap() as u16)
    {
        hash = 31i32.wrapping_mul(hash).wrapping_add(b as i32);
    }
    hash
}

/// A lerp (Linear Interpolation)
pub fn lerp(v0: f64, v1: f64, t: f64) -> f64 {
    v0 + (t * (v1 - v0))
}

/// A lerp (Linear Interpolation) where `t` is clamped between 0 and 1,
/// and the result is therefore clamped between `v0` and `v1`.
pub fn clamped_lerp(v0: f64, v1: f64, t: f64) -> f64 {
    if t <= 0.0 {
        v0
    } else if t >= 1.0 {
        v1
    } else {
        lerp(v0, v1, t)
    }
}

/// A lerp (Linear Interpolation)
pub fn lerp_f32(v0: f32, v1: f32, t: f32) -> f32 {
    v0 + (t * (v1 - v0))
}

/// A bilinear interpolation (2d lerp)
///
/// The different parameters represent the coordinates for t where the output would be that parameter.
/// e.g. `lerp2(..., DVec2::new(0., 0.))` would output `v00`, `DVec2::new(1., 0.)` would output `v10`
pub fn lerp2(v00: f64, v10: f64, v01: f64, v11: f64, t: DVec2) -> f64 {
    lerp(lerp(v00, v10, t.x), lerp(v01, v11, t.x), t.y)
}

/// A trilinear interpolation (3d lerp)
///
/// The different parameters represent the coordinates for t where the output would be that parameter.
/// e.g. `lerp3(..., DVec3::new(0., 0., 0.))` would output `v000`, `DVec3::new(1., 0., 1.)` would output `v101`
#[allow(clippy::too_many_arguments)]
pub fn lerp3(
    v000: f64,
    v100: f64,
    v010: f64,
    v110: f64,
    v001: f64,
    v101: f64,
    v011: f64,
    v111: f64,
    t: DVec3,
) -> f64 {
    lerp(
        lerp2(v000, v100, v010, v110, t.xy()),
        lerp2(v001, v101, v011, v111, t.xy()),
        t.z,
    )
}

pub fn smoothstep(v: f64) -> f64 {
    v * v * v * (v * (v * 6.0 - 15.0) + 10.0)
}

/// Generates a seed for RNGs from a position
pub fn get_seed(pos: IVec3) -> i64 {
    let mut val = (pos.x.wrapping_mul(3129871) as i64)
        ^ 116129781i64.wrapping_mul(pos.z as i64)
        ^ pos.y as i64;
    val = val
        .wrapping_mul(val)
        .wrapping_mul(42317861)
        .wrapping_add(val.wrapping_mul(11));
    val >> 16
}

pub fn floor_i32(value: f64) -> i32 {
    let int = value as i32;
    if value < int as f64 { int - 1 } else { int }
}

pub fn floor_i64(value: f64) -> i64 {
    let i = value as i64;
    if value < i as f64 { i - 1 } else { i }
}

pub fn transmute_to_i32(value: u32) -> i32 {
    i32::from_ne_bytes(value.to_ne_bytes())
}

pub fn transmute_to_u64(value: i64) -> u64 {
    u64::from_ne_bytes(value.to_ne_bytes())
}

pub fn transmute_to_i64(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

pub fn u128_seed_from_hash_of(text: &str) -> (u64, u64) {
    let digest = md5::compute(text).0;
    (
        u64::from_be_bytes(digest[0..8].try_into().unwrap()),
        u64::from_be_bytes(digest[8..16].try_into().unwrap()),
    )
}

#[cfg(all(test, feature = "java_tests_module"))]
mod java_tests {
    use jni::objects::JValue;

    use crate::{
        helpers::{self, transmute_to_u64},
        java_tests::{Env, Mth, RandomSupport, Seed128bit, get_jvm_env},
    };

    #[test]
    fn floor() {
        let mut env = get_jvm_env();
        for value in [0.0, -1.0, -1.4, -1.6, 2.0, 1.5, 1.6, 1.3] {
            assert_eq!(
                env.call_static(&Mth::FLOOR, &[JValue::Double(value)])
                    .i()
                    .unwrap(),
                super::floor_i32(value),
                "value = {value}"
            );
        }
    }

    #[test]
    fn smoothstep() {
        let mut env = get_jvm_env();
        for value in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0] {
            assert_eq!(
                env.call_static(&Mth::SMOOTHSTEP, &[JValue::Double(value)])
                    .d()
                    .unwrap(),
                helpers::smoothstep(value),
                "value = {value}"
            );
        }
    }

    #[test]
    fn u128_seed_from_hash_of() {
        let mut env = get_jvm_env();
        assert_eq!(
            java_seed_from_hash_of(&mut env, ""),
            super::u128_seed_from_hash_of("")
        );
        assert_eq!(
            java_seed_from_hash_of(&mut env, "abcdef"),
            super::u128_seed_from_hash_of("abcdef")
        );
        assert_eq!(
            java_seed_from_hash_of(&mut env, "✅"),
            super::u128_seed_from_hash_of("✅")
        );
        assert_eq!(
            java_seed_from_hash_of(&mut env, "abc✅❌🚧👌abc"),
            super::u128_seed_from_hash_of("abc✅❌🚧👌abc")
        );
    }

    fn java_seed_from_hash_of(env: &mut Env, input: &str) -> (u64, u64) {
        let input = env.string(input);
        let seed128 = env
            .call_static(
                &RandomSupport::SEED_FROM_HASH_OF,
                &[JValue::Object(&input.into())],
            )
            .l()
            .unwrap();
        let lo = env.field(&seed128, &Seed128bit::SEED_LO).j().unwrap();
        let hi = env.field(&seed128, &Seed128bit::SEED_HI).j().unwrap();
        (transmute_to_u64(lo), transmute_to_u64(hi))
    }
}

#[cfg(test)]
mod tests {
    use glam::{DVec2, DVec3, IVec3};

    use crate::helpers;

    #[test]
    fn java_str_hash() {
        assert_eq!(helpers::java_str_hash(""), 0);
        assert_eq!(helpers::java_str_hash("abcdef"), -1424385949);
        assert_eq!(helpers::java_str_hash("✅"), 9989);
        assert_eq!(helpers::java_str_hash("abc✅❌🚧👌abc"), -231361120);
    }

    #[test]
    fn lerp() {
        assert_eq!(helpers::lerp(100.0, 50.0, 0.0), 100.0);
        assert_eq!(helpers::lerp(100.0, 50.0, 1.0), 50.0);
    }

    #[test]
    fn lerp2() {
        assert_eq!(
            helpers::lerp2(1.0, 2.0, 3.0, 4.0, DVec2::new(0.0, 0.0)),
            1.0
        );
        assert_eq!(
            helpers::lerp2(1.0, 2.0, 3.0, 4.0, DVec2::new(1.0, 0.0)),
            2.0
        );
        assert_eq!(
            helpers::lerp2(1.0, 2.0, 3.0, 4.0, DVec2::new(0.0, 1.0)),
            3.0
        );
        assert_eq!(
            helpers::lerp2(1.0, 2.0, 3.0, 4.0, DVec2::new(1.0, 1.0)),
            4.0
        );
    }

    #[test]
    fn lerp3() {
        for i in 0..8 {
            let x = i & 1;
            let y = (i >> 1) & 1;
            let z = (i >> 2) & 1;
            dbg!(x, y, z);
            assert_eq!(
                helpers::lerp3(
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    DVec3::new(x as f64, y as f64, z as f64)
                ),
                i as f64
            );
        }
    }

    #[test]
    fn get_seed() {
        assert_eq!(helpers::get_seed(IVec3::new(0, 0, 0)), 0);
        assert_eq!(helpers::get_seed(IVec3::new(1, 2, 3)), -33674130277896);
        assert_eq!(helpers::get_seed(IVec3::new(-8, -5, 37)), 8594576500896);
    }
}
