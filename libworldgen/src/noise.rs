use glam::{DVec2, DVec3, IVec2, IVec3};
use mcpackloader::worldgen::NoiseParameters;

use crate::{
    helpers::{floor_i32, floor_i64, lerp3, smoothstep},
    random::{Rng, RngFactory},
};

/// Noise used in density functions
/// Equivalent to `net.minecraft.world.level.levelgen.synth.NormalNoise`
pub struct NormalNoise {
    value_factor: f64,
    first: PerlinNoise,
    second: PerlinNoise,
    max_value: f64,
    parameters: NoiseParameters,
}

impl NormalNoise {
    /// Creates a new [`NormalNoise`] from some [`NoiseParameters`].
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
    /// Equivalent to `false` in the java code
    Legacy,
    /// Equivalent to `true` in the java code
    New,
}

impl PerlinNoise {
    pub fn create(rng: &impl Rng, first_octave: i32, amplitudes: Vec<f64>) -> Self {
        Self::new(rng, first_octave, amplitudes, PerlinNoiseMode::New)
    }

    fn new(rng: &impl Rng, first_octave: i32, amplitudes: Vec<f64>, mode: PerlinNoiseMode) -> Self {
        let ampls = amplitudes.len() as i32;
        let neg_first_octave = -first_octave;
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

    pub fn wrap(value: f64) -> f64 {
        value - floor_i64(value / 3.3554432e7 + 0.5) as f64 * 3.3554432e7
    }

    /// Creates a new [`PerlinNoise`] for use by [`BlendedNoise`].
    ///
    /// Takes in a *sorted* list of octaves.
    ///
    /// # Panics
    ///
    /// - if `octaves` is empty
    /// - if octaves isn't sorted in ascending order (may only panic sometimes)
    ///
    /// [`BlendedNoise`]: crate::density_function::ops::BlendedNoise
    pub fn create_for_blended_noise(rng: &impl Rng, octaves: &[i32]) -> Self {
        let (first_octave, amplitudes) = Self::make_amplitudes(octaves);
        Self::new(rng, first_octave, amplitudes, PerlinNoiseMode::Legacy)
    }

    /// Generates amplitudes from a list of octaves, used for [`Self::create_for_blended_noise`].
    ///
    /// Takes in a *sorted* list of octaves.
    ///
    /// # Panics
    ///
    /// - if `octaves` is empty
    /// - if octaves isn't sorted in ascending order (may only panic sometimes)
    fn make_amplitudes(octaves: &[i32]) -> (i32, Vec<f64>) {
        assert!(!octaves.is_empty(), "`octaves` shouldn't be empty");

        let first_octave = *octaves.first().unwrap();
        let octave_range = octaves.last().unwrap() - first_octave + 1;
        assert!(
            octave_range > 0,
            "`octaves` should be sorted in ascending order"
        );

        let mut amplitudes = vec![0.0; octave_range as usize];
        for octave in octaves {
            assert!(
                octave - first_octave >= 0,
                "`octaves` should be sorted in ascending order"
            );
            amplitudes[(octave - first_octave) as usize] = 1.0;
        }

        (first_octave, amplitudes)
    }

    /// Gets the [`ImprovedNoise`] instance of an octave.
    pub fn get_octave_noise(&self, octave: i32) -> Option<&ImprovedNoise> {
        let idx = self.noise_levels.len() as i32 - 1 - octave;
        // check cast to usize
        if idx < 0 {
            return None;
        }
        self.noise_levels.get(idx as usize).and_then(Option::as_ref)
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

    fn sample_and_lerp(
        &self,
        rounded_pos: IVec3,
        remainder_pos: DVec3,
        raw_remainder_y: f64,
    ) -> f64 {
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
        let ly = smoothstep(raw_remainder_y);
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

    /// Underlying generation method, prefer [`ImprovedNoise::get`] instead.
    /// This method is only exposed for [`BlendedNoise`].
    ///
    /// [`BlendedNoise`]: crate::density_function::ops::BlendedNoise
    pub fn get_smeared(&self, pos: DVec3, smeared_y: f64, raw_y: f64) -> f64 {
        let pos = self.offset + pos;
        let rounded_pos = IVec3::new(floor_i32(pos.x), floor_i32(pos.y), floor_i32(pos.z));
        let remainder_pos = pos - rounded_pos.as_dvec3();
        let y_mod = if smeared_y != 0.0 {
            // FIXME better variable name for d7
            let d7 = if raw_y >= 0.0 && raw_y < remainder_pos.y {
                raw_y
            } else {
                remainder_pos.y
            };

            floor_i32(d7 / smeared_y + 1e-7) as f64 * smeared_y
        } else {
            0.0
        };
        self.sample_and_lerp(
            rounded_pos,
            remainder_pos.with_y(remainder_pos.y - y_mod),
            remainder_pos.y,
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
        self.get_smeared(pos, 0.0, 0.0)
    }
}

/// A 3d noise
pub trait Noise3d {
    /// Samples the noise at a given location
    fn get(&self, pos: DVec3) -> f64;
}

pub struct SimplexNoise {
    lut: [u8; 256],
}

impl SimplexNoise {
    pub fn new(rng: &impl Rng) -> Self {
        // used to generate xo, yo, zo, but these aren't used
        rng.next_f64();
        rng.next_f64();
        rng.next_f64();

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

        Self { lut }
    }

    pub fn get_2d(&self, pos: DVec2) -> f64 {
        // oh god!?
        let pos_sum = pos.element_sum() * Self::F2;
        let floor_pos = IVec2::new(floor_i32(pos.x + pos_sum), floor_i32(pos.y + pos_sum));
        let floor_sum = floor_pos.element_sum() as f64 * Self::G2;
        let floor_dev = floor_pos.as_dvec2() - floor_sum;
        let corner1 = pos - floor_dev;
        let k = if corner1.x > corner1.y {
            IVec2::new(1, 0)
        } else {
            IVec2::new(0, 1)
        };
        let corner2 = corner1 - k.as_dvec2() + Self::G2;
        let corner3 = corner1 - 1.0 + 2.0 * Self::G2;
        let i = floor_pos & 255;
        let grad1 = self.lut(i.x + self.lut(i.y)) % 12;
        let grad2 = self.lut(i.x + k.x + self.lut(i.y + k.y)) % 12;
        let grad3 = self.lut(i.x + 1 + self.lut(i.y + 1)) % 12;
        let noise1 =
            Self::get_corner_noise_3d(grad1 as usize, DVec3::new(corner1.x, corner1.y, 0.0), 0.5);
        let noise2 =
            Self::get_corner_noise_3d(grad2 as usize, DVec3::new(corner2.x, corner2.y, 0.0), 0.5);
        let noise3 =
            Self::get_corner_noise_3d(grad3 as usize, DVec3::new(corner3.x, corner3.y, 0.0), 0.5);
        70.0 * (noise1 + noise2 + noise3)
    }

    fn lut(&self, i: i32) -> i32 {
        self.lut[i as usize & 255] as i32
    }

    fn get_corner_noise_3d(i: usize, pos: DVec3, min: f64) -> f64 {
        let dist_sq = min - pos.x.powi(2) - pos.y.powi(2) - pos.z.powi(2);
        if dist_sq < 0.0 {
            0.0
        } else {
            dist_sq.powi(4) * (Self::GRADIENT[i].as_dvec3() * pos).element_sum()
        }
    }

    const SQRT_3: f64 = 1.7320508075688772;
    const F2: f64 = 0.5 * (Self::SQRT_3 - 1.0);
    const G2: f64 = (3.0 - Self::SQRT_3) / 6.0;

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
    use glam::{DVec2, DVec3};
    use jni::objects::JValue;

    use crate::{
        java_tests::{self, Class, DoubleArrayList, get_jvm_env},
        noise::{ImprovedNoise, Noise3d, PerlinNoise, SimplexNoise},
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
    fn perlin_noise_wrap() {
        let mut env = get_jvm_env();
        for i in -10..10 {
            let i = i as f64 / 4.0;
            assert_eq!(
                env.call_static(&java_tests::PerlinNoise::WRAP, &[JValue::Double(i)])
                    .d()
                    .unwrap(),
                PerlinNoise::wrap(i),
                "i = {i}"
            )
        }
    }

    // normal noise isn't tested, as it's constructor requires `NoiseParameters`,
    // which relies on registries. Idk how to set those up.

    #[test]
    fn perlin_noise() {
        let mut env = get_jvm_env();
        let java_rng = env.construct(
            &Class::LegacyRandomSource,
            java_tests::RandomSource::CONSTRUCTOR,
            &[JValue::Long(0)],
        );
        let java_amplitudes = DoubleArrayList::create(&mut env, &[3.0, 2.0, 1.0]);
        let java_perlin_noise = env
            .call_static(
                &java_tests::PerlinNoise::CREATE,
                &[
                    JValue::Object(&java_rng),
                    JValue::Int(2),
                    JValue::Object(&java_amplitudes),
                ],
            )
            .l()
            .unwrap();
        let rng = LegacyRng::new(0);
        let perlin_noise = PerlinNoise::create(&rng, 2, vec![3.0, 2.0, 1.0]);
        for x in -5..5 {
            for y in -5..5 {
                for z in -5..5 {
                    let x = x as f64 / 2.5;
                    let y = y as f64 / 2.5;
                    let z = z as f64 / 2.5;

                    assert_eq!(
                        env.call(
                            &java_perlin_noise,
                            &java_tests::PerlinNoise::GET_VALUE,
                            &[JValue::Double(x), JValue::Double(y), JValue::Double(z)]
                        )
                        .d()
                        .unwrap(),
                        perlin_noise.get(DVec3::new(x, y, z)),
                        "at {x}, {y}, {z}"
                    );
                }
            }
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
                    let x = x as f64 / 2.5;
                    let y = y as f64 / 2.5;
                    let z = z as f64 / 2.5;
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

    #[test]
    fn simplex_noise() {
        let mut env = get_jvm_env();
        let java_rng = env.construct(
            &Class::LegacyRandomSource,
            java_tests::RandomSource::CONSTRUCTOR,
            &[JValue::Long(0)],
        );
        let java_simplex_noise = env.construct(
            &Class::SimplexNoise,
            java_tests::SimplexNoise::CONSTRUCTOR,
            &[JValue::Object(&java_rng)],
        );
        let rng = LegacyRng::new(0);
        let simplex_noise = SimplexNoise::new(&rng);

        for x in -10..10 {
            for y in -10..10 {
                let x = x as f64 / 2.5;
                let y = y as f64 / 2.5;
                assert_more_or_less_eq!(
                    env.call(
                        &java_simplex_noise,
                        &java_tests::SimplexNoise::GET_VALUE_2D,
                        &[JValue::Double(x), JValue::Double(y)]
                    )
                    .d()
                    .unwrap(),
                    simplex_noise.get_2d(DVec2::new(x, y)),
                    "at {x}, {y}"
                );
            }
        }
    }
}
