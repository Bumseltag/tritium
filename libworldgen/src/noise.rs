use glam::{DVec3, IVec3};
use mcpackloader::worldgen::NoiseParameters;

use crate::{
    helpers::{floor_i32, floor_i64, lerp3, smoothstep},
    random::{Rng, RngFactory},
};

pub struct NormalNoise {
    value_factor: f64,
    first: PerlinNoise,
    second: PerlinNoise,
    max_value: f64,
    parameters: NoiseParameters,
}

impl NormalNoise {
    pub fn new(rng: &impl Rng, parameters: NoiseParameters) -> Self {
        let first =
            PerlinNoise::create(rng, parameters.first_octave, parameters.amplitudes.clone());
        let second =
            PerlinNoise::create(rng, parameters.first_octave, parameters.amplitudes.clone());

        let mut min_nonzero_octave = i32::MAX;
        let mut max_nonzero_octave = i32::MIN;
        for (i, amplitude) in parameters.amplitudes.iter().enumerate() {
            if *amplitude != 0.0 {
                min_nonzero_octave = min_nonzero_octave.min(i as i32);
                max_nonzero_octave = max_nonzero_octave.max(i as i32);
            }
        }

        let value_factor =
            0.16666666666666666 / Self::expected_deviation(max_nonzero_octave - min_nonzero_octave);
        let max_value = (first.max_value + second.max_value) * value_factor;
        Self {
            value_factor,
            first,
            second,
            max_value,
            parameters,
        }
    }

    fn expected_deviation(octave_range: i32) -> f64 {
        0.1 * (1.0 + (1.0 / (octave_range + 1) as f64))
    }
}

impl Noise3d for NormalNoise {
    fn get(&self, pos: DVec3) -> f64 {
        (self.second.get(pos * 1.0181268882175227) + self.first.get(pos)) * self.value_factor
    }
}

/// Equivalent to `net.minecraft.world.level.levelgen.synth.PerlinNoise`
pub struct PerlinNoise {
    noise_levels: Vec<Option<ImprovedNoise>>,
    first_octave: i32,
    amplitudes: Vec<f64>,
    lowest_freq_value_factor: f64,
    lowest_freq_input_factor: f64,
    max_value: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PerlinNoiseMode {
    Legacy,
    New,
}

impl PerlinNoise {
    pub fn create(rng: &impl Rng, first_octave: i32, amplitudes: Vec<f64>) -> Self {
        Self::new(rng, first_octave, amplitudes, PerlinNoiseMode::New)
    }

    fn new(rng: &impl Rng, first_octave: i32, amplitudes: Vec<f64>, mode: PerlinNoiseMode) -> Self {
        let ampls = amplitudes.len() as i32;
        let neg_first_octave = first_octave;
        let mut noise_levels = vec![None; amplitudes.len()];
        if mode == PerlinNoiseMode::New {
            let rng_factory = rng.fork_factory();

            for (i, amplitude) in amplitudes.iter().enumerate() {
                if *amplitude != 0.0 {
                    let octave = first_octave + i as i32;
                    noise_levels[i] = Some(ImprovedNoise::new(
                        &rng_factory.hash_of(&format!("octave_{octave}")),
                    ));
                }
            }
        } else {
            let first_octave_noise = ImprovedNoise::new(rng);
            if neg_first_octave >= 0 && neg_first_octave < ampls {
                let amplitude = amplitudes[neg_first_octave as usize];
                if amplitude != 0.0 {
                    noise_levels[neg_first_octave as usize] = Some(first_octave_noise);
                }
            }
            for i in (0..=(neg_first_octave - 1)).rev() {
                if i < ampls {
                    let amplitude = amplitudes[i as usize];
                    if amplitude != 0.0 {
                        noise_levels[i as usize] = Some(ImprovedNoise::new(rng));
                    } else {
                        rng.skip_i32(262);
                    }
                } else {
                    rng.skip_i32(262);
                }
            }
        }

        let lowest_freq_input_factor = 2.0f64.powi(first_octave);
        let lowest_freq_value_factor = 2.0f64.powi(ampls - 1) / (2.0f64.powi(ampls) - 1.0);

        let mut max_value = 0.0;
        let mut value_factor = lowest_freq_value_factor;
        for (i, noise) in noise_levels.iter().enumerate() {
            if noise.is_some() {
                max_value += amplitudes[i] * value_factor;
            }

            value_factor /= 2.0;
        }
        max_value *= 2.0;

        Self {
            noise_levels,
            first_octave,
            amplitudes,
            lowest_freq_input_factor,
            lowest_freq_value_factor,
            max_value,
        }
    }

    fn wrap(value: f64) -> f64 {
        value - floor_i64(value / 3.3554432e7 + 0.5) as f64 * 3.3554432e7
    }
}

impl Noise3d for PerlinNoise {
    fn get(&self, pos: DVec3) -> f64 {
        let mut res = 0.0;
        let mut input_factor = self.lowest_freq_input_factor;
        let mut value_factor = self.lowest_freq_value_factor;

        for (i, noise) in self.noise_levels.iter().enumerate() {
            if let Some(noise) = noise {
                let noise_val = noise.get(DVec3::new(
                    Self::wrap(pos.x * input_factor),
                    Self::wrap(pos.y * input_factor),
                    Self::wrap(pos.z * input_factor),
                ));
                res += self.amplitudes[i] * noise_val * value_factor;
            }

            input_factor *= 2.0;
            value_factor /= 2.0;
        }

        res
    }
}

/// Equivalent to `net.minecraft.world.level.levelgen.synth.ImprovedNoise`
#[derive(Debug, Clone)]
pub struct ImprovedNoise {
    lut: [u8; 256],
    offset: DVec3,
}

impl ImprovedNoise {
    pub fn new(rng: &impl Rng) -> Self {
        let x = rng.next_f64() * 256.0;
        let y = rng.next_f64() * 256.0;
        let z = rng.next_f64() * 256.0;

        // generate initial lut [0, 1, 2, 3, ...]
        let mut lut = const {
            let mut lut = [0u8; 256];
            let mut i = 0;
            while i < 256 {
                lut[i] = i as u8;
                i += 1;
            }
            lut
        };

        // shuffle lut
        for i in 0..256 {
            let j = rng.next_i32_up_to(256 - i as i32) as usize + i;
            lut.swap(i, j);
        }

        Self {
            lut,
            offset: DVec3::new(x, y, z),
        }
    }

    fn sample_and_lerp(&self, rounded_pos: IVec3, remainder_pos: DVec3) -> f64 {
        // im so sorry for whats about to happen but i can't think of better variable names
        let x = self.lut(rounded_pos.x);
        let x1 = self.lut(rounded_pos.x + 1);
        let a = self.lut(x + rounded_pos.y);
        let a1 = self.lut(x + rounded_pos.y + 1);
        let b = self.lut(x1 + rounded_pos.y);
        let b1 = self.lut(x1 + rounded_pos.y + 1);
        let v000 = Self::grad_dot(self.lut(a + rounded_pos.z), remainder_pos);
        let v100 = Self::grad_dot(
            self.lut(b + rounded_pos.z),
            remainder_pos - DVec3::new(1.0, 0.0, 0.0),
        );
        let v010 = Self::grad_dot(
            self.lut(a1 + rounded_pos.z),
            remainder_pos - DVec3::new(0.0, 1.0, 0.0),
        );
        let v110 = Self::grad_dot(
            self.lut(b1 + rounded_pos.z),
            remainder_pos - DVec3::new(1.0, 1.0, 0.0),
        );
        let v001 = Self::grad_dot(
            self.lut(a + rounded_pos.z + 1),
            remainder_pos - DVec3::new(0.0, 0.0, 1.0),
        );
        let v101 = Self::grad_dot(
            self.lut(b + rounded_pos.z + 1),
            remainder_pos - DVec3::new(1.0, 0.0, 1.0),
        );
        let v011 = Self::grad_dot(
            self.lut(a1 + rounded_pos.z + 1),
            remainder_pos - DVec3::new(0.0, 1.0, 1.0),
        );
        let v111 = Self::grad_dot(
            self.lut(b1 + rounded_pos.z + 1),
            remainder_pos - DVec3::new(1.0, 1.0, 1.0),
        );
        let lx = smoothstep(remainder_pos.x);
        let ly = smoothstep(remainder_pos.y);
        let lz = smoothstep(remainder_pos.z);
        lerp3(
            v000,
            v100,
            v010,
            v110,
            v001,
            v101,
            v011,
            v111,
            DVec3::new(lx, ly, lz),
        )
    }

    fn lut(&self, i: i32) -> i32 {
        self.lut[i as usize & 255] as i32
    }

    fn grad_dot(i: i32, pos: DVec3) -> f64 {
        (pos * SimplexNoise::GRADIENT[(i & 15) as usize].as_dvec3()).element_sum()
    }
}

impl Noise3d for ImprovedNoise {
    fn get(&self, pos: DVec3) -> f64 {
        let pos = self.offset + pos;
        let rounded_pos = IVec3::new(floor_i32(pos.x), floor_i32(pos.y), floor_i32(pos.z));
        let remainder_pos = pos - rounded_pos.as_dvec3();
        self.sample_and_lerp(rounded_pos, remainder_pos)
    }
}

pub trait Noise3d {
    fn get(&self, pos: DVec3) -> f64;
}

pub struct SimplexNoise {}

impl SimplexNoise {
    const GRADIENT: [IVec3; 16] = [
        IVec3::new(1, 1, 0),
        IVec3::new(-1, 1, 0),
        IVec3::new(1, -1, 0),
        IVec3::new(-1, -1, 0),
        IVec3::new(1, 0, 1),
        IVec3::new(-1, 0, 1),
        IVec3::new(1, 0, -1),
        IVec3::new(-1, 0, -1),
        IVec3::new(0, 1, 1),
        IVec3::new(0, -1, 1),
        IVec3::new(0, 1, -1),
        IVec3::new(0, -1, -1),
        IVec3::new(1, 1, 0),
        IVec3::new(0, -1, 1), // why ??
        IVec3::new(-1, 1, 0),
        IVec3::new(0, -1, -1),
    ];
}

#[cfg(all(test, feature = "java_tests_module"))]
mod java_tests {
    use glam::DVec3;
    use jni::objects::JValue;

    use crate::{
        java_tests::{self, Class, get_jvm_env},
        noise::{ImprovedNoise, Noise3d},
        random::LegacyRng,
    };

    #[test]
    fn grad_dot() {
        let mut env = get_jvm_env();
        for i in [0, 15, 123] {
            assert_eq!(
                env.call_static(
                    &java_tests::ImprovedNoise::GRAD_DOT,
                    &[
                        JValue::Int(0),
                        JValue::Double(10.0),
                        JValue::Double(-10.0),
                        JValue::Double(123.0)
                    ]
                )
                .d()
                .unwrap(),
                ImprovedNoise::grad_dot(0, DVec3::new(10.0, -10.0, 123.0)),
                "i = {i}"
            );
        }
    }

    #[test]
    fn improved_noise_constructor() {
        let mut env = get_jvm_env();
        let java_rng = env.construct(
            &Class::LegacyRandomSource,
            java_tests::RandomSource::CONSTRUCTOR,
            &[JValue::Long(0)],
        );
        let java_improved_noise = env.construct(
            &Class::ImprovedNoise,
            java_tests::ImprovedNoise::CONSTRUCTOR,
            &[JValue::Object(&java_rng)],
        );
        let rng = LegacyRng::new(0);
        let improved_noise = ImprovedNoise::new(&rng);
        for i in 0..20 {
            assert_eq!(
                env.call(
                    &java_improved_noise,
                    &java_tests::ImprovedNoise::P,
                    &[JValue::Int(i)]
                )
                .i()
                .unwrap(),
                improved_noise.lut(i),
                "i = {i}"
            );
        }
        assert_eq!(
            env.field(&java_improved_noise, &java_tests::ImprovedNoise::XO)
                .d()
                .unwrap(),
            improved_noise.offset.x
        );
        assert_eq!(
            env.field(&java_improved_noise, &java_tests::ImprovedNoise::YO)
                .d()
                .unwrap(),
            improved_noise.offset.y
        );
        assert_eq!(
            env.field(&java_improved_noise, &java_tests::ImprovedNoise::ZO)
                .d()
                .unwrap(),
            improved_noise.offset.z
        );
    }

    #[test]
    fn improved_noise() {
        let mut env = get_jvm_env();
        let java_rng = env.construct(
            &Class::LegacyRandomSource,
            java_tests::RandomSource::CONSTRUCTOR,
            &[JValue::Long(0)],
        );
        let java_improved_noise = env.construct(
            &Class::ImprovedNoise,
            java_tests::ImprovedNoise::CONSTRUCTOR,
            &[JValue::Object(&java_rng)],
        );
        let rng = LegacyRng::new(0);
        let improved_noise = ImprovedNoise::new(&rng);

        for x in -5..5 {
            for y in -5..5 {
                for z in -5..5 {
                    let x = x as f64 / 5.0;
                    let y = y as f64 / 5.0;
                    let z = z as f64 / 5.0;
                    assert_eq!(
                        env.call(
                            &java_improved_noise,
                            &java_tests::ImprovedNoise::NOISE,
                            &[JValue::Double(x), JValue::Double(y), JValue::Double(z)]
                        )
                        .d()
                        .unwrap(),
                        improved_noise.get(DVec3::new(x, y, z)),
                        "at {x}, {y}, {z}"
                    );
                }
            }
        }
    }
}
