//! The standard Minecraft density function operations

use std::{
    ops::{Add as _, BitAnd, Mul as _},
    sync::LazyLock,
};

use glam::{DVec3, I64Vec3, IVec2, Vec3Swizzles};
use mcpackloader::{ResourceLocation, worldgen::NoiseParameters};

use crate::{
    density_function::{FunctionOp, FunctionOpType, NonDynFunctionOp, SUBCHUNK_SIZE},
    helpers::{clamped_lerp, lerp, lerp3},
    noise::{Noise3d, NormalNoise, PerlinNoise, SimplexNoise},
    random::{LegacyRng, Rng, XoroshiroRng},
    spline,
};

/// Rounds the number to the nearest lower multiple of 4.
///
/// This function is generic and therefore works with i64 but also [`I64Vec3`] and similar types.
fn floor4<T: BitAnd<i64>>(v: T) -> T::Output {
    v & (-1 ^ 3/* = 0b1111...1111100 */)
}

pub struct Interpolated(pub Box<dyn FunctionOp>);

impl FunctionOp for Interpolated {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        let floor = floor4(pos);
        let floor1 = floor + 4;
        let rem = pos & 3;
        let v000 = self.0.run_once(&floor);
        let v100 = self.0.run_once(&I64Vec3::new(floor1.x, floor.y, floor.z));
        let v010 = self.0.run_once(&I64Vec3::new(floor.x, floor1.y, floor.z));
        let v110 = self.0.run_once(&I64Vec3::new(floor1.x, floor1.y, floor.z));
        let v001 = self.0.run_once(&I64Vec3::new(floor.x, floor.y, floor1.z));
        let v101 = self.0.run_once(&I64Vec3::new(floor1.x, floor.y, floor1.z));
        let v011 = self.0.run_once(&I64Vec3::new(floor.x, floor1.y, floor1.z));
        let v111 = self.0.run_once(&I64Vec3::new(floor1.x, floor1.y, floor1.z));
        lerp3(
            v000,
            v100,
            v010,
            v110,
            v001,
            v101,
            v011,
            v111,
            rem.as_dvec3() / 4.0,
        )
    }

    fn run_plane(&self, pos: &I64Vec3) -> [f64; 16 * 16] {
        let y_floor = floor4(pos.y);
        let y_floor1 = y_floor + 4;
        let y_rem = (pos.y & 3) as f64 / 4.0;
        let mut cells = [f64::NAN; 5 * 5];
        for cx in 0..5 {
            for cz in 0..5 {
                cells[cx + (cz * 5)] = lerp(
                    self.0.run_once(&I64Vec3::new(
                        pos.x + (cx as i64 * 4),
                        y_floor,
                        pos.z + (cz as i64 * 4),
                    )),
                    self.0.run_once(&I64Vec3::new(
                        pos.x + (cx as i64 * 4),
                        y_floor1,
                        pos.z + (cz as i64 * 4),
                    )),
                    y_rem,
                );
            }
        }
        let mut res = [f64::NAN; 16 * 16];
        for cx in 0..4 {
            for cz in 0..4 {
                let v0x0 = cells[cx + (cz * 5)];
                let v1x0 = cells[cx + (cz * 5) + 1];
                let v0x1 = cells[cx + (cz * 5) + 5];
                let v1x1 = cells[cx + (cz * 5) + 1 + 5];
                let x_offset = cx * 4;
                let z_offset = cz * 4;
                for x in 0..4 {
                    let x_rem = x as f64 / 4.0;
                    let vxx0 = lerp(v0x0, v1x0, x_rem);
                    let vxx1 = lerp(v0x1, v1x1, x_rem);
                    for z in 0..4 {
                        let z_rem = z as f64 / 4.0;
                        res[(x + x_offset) + ((z + z_offset) * 16)] = lerp(vxx0, vxx1, z_rem);
                    }
                }
            }
        }
        res
    }

    fn run_subchunk(&self, pos: &I64Vec3) -> [f64; SUBCHUNK_SIZE] {
        let mut cells = [f64::NAN; 5 * 5 * 5];
        for cx in 0..5 {
            for cz in 0..5 {
                for cy in 0..5 {
                    cells[cx + (cz * 5) + (cy * 5 * 5)] = self.0.run_once(&I64Vec3::new(
                        pos.x + (cx as i64 * 4),
                        pos.y + (cy as i64 * 4),
                        pos.z + (cz as i64 * 4),
                    ));
                }
            }
        }
        let mut res = [f64::NAN; SUBCHUNK_SIZE];
        for cy in 0..4 {
            for cx in 0..4 {
                for cz in 0..4 {
                    let i = cx + (cz * 5) + (cy * 5 * 5);
                    let v000 = cells[i];
                    let v100 = cells[i + 1];
                    let v010 = cells[i + 25];
                    let v110 = cells[i + 1 + 25];
                    let v001 = cells[i + 5];
                    let v101 = cells[i + 1 + 5];
                    let v011 = cells[i + 25 + 5];
                    let v111 = cells[i + 1 + 25 + 5];
                    let x_offset = cx * 4;
                    let y_offset = cy * 4;
                    let z_offset = cz * 4;
                    for y in 0..4 {
                        let y_rem = y as f64 / 4.0;
                        let v0x0 = lerp(v000, v010, y_rem);
                        let v1x0 = lerp(v100, v110, y_rem);
                        let v0x1 = lerp(v001, v011, y_rem);
                        let v1x1 = lerp(v101, v111, y_rem);
                        for x in 0..4 {
                            let x_rem = x as f64 / 4.0;
                            let vxx0 = lerp(v0x0, v1x0, x_rem);
                            let vxx1 = lerp(v0x1, v1x1, x_rem);
                            for z in 0..4 {
                                let z_rem = z as f64 / 4.0;
                                res[(x + x_offset)
                                    + ((z + z_offset) * 16)
                                    + ((y + y_offset) * 16 * 16)] = lerp(vxx0, vxx1, z_rem);
                            }
                        }
                    }
                }
            }
        }
        res
    }
}

pub struct FlatCache(pub Box<dyn FunctionOp>);

impl FunctionOp for FlatCache {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        self.0.run_once(&floor4(pos).with_y(0))
    }

    fn run_plane(&self, pos: &I64Vec3) -> [f64; 16 * 16] {
        let mut res = [f64::NAN; 16 * 16];
        for cx in 0..4 {
            for cz in 0..4 {
                let val = self
                    .0
                    .run_once(&(pos.with_y(0) + I64Vec3::new(cx * 4, 0, cz * 4)));
                for z in (cz * 4)..(cz * 4 + 4) {
                    let idx = (cx as usize * 4) + (z as usize * 16);
                    res[idx..(idx + 4)].copy_from_slice(&[val; 4]);
                }
            }
        }
        res
    }

    fn run_subchunk(&self, pos: &I64Vec3) -> [f64; SUBCHUNK_SIZE] {
        self.run_plane(pos).repeat(16).try_into().unwrap()
    }
}

pub struct Cache2d(Box<dyn FunctionOp>);

impl FunctionOp for Cache2d {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        self.0.run_once(pos)
    }

    fn run_plane(&self, pos: &I64Vec3) -> [f64; 16 * 16] {
        self.0.run_plane(pos)
    }

    fn run_subchunk(&self, pos: &I64Vec3) -> [f64; SUBCHUNK_SIZE] {
        self.0
            .run_plane(&pos.with_y(0))
            .repeat(16)
            .try_into()
            .unwrap()
    }
}

pub struct BlendedNoise {
    min_limit_noise: PerlinNoise,
    max_limit_noise: PerlinNoise,
    main_noise: PerlinNoise,
    xz_multiplier: f64,
    y_multiplier: f64,
    xz_factor: f64,
    y_factor: f64,
    smear_scale_multiplier: f64,
    xz_scale: f64,
    y_scale: f64,
}

impl BlendedNoise {
    pub fn create_unseeded(
        xz_scale: f64,
        y_scale: f64,
        xz_factor: f64,
        y_factor: f64,
        smear_scale_multiplier: f64,
    ) -> Self {
        Self::new(
            &XoroshiroRng::new(0),
            xz_scale,
            y_scale,
            xz_factor,
            y_factor,
            smear_scale_multiplier,
        )
    }

    fn new(
        rng: &impl Rng,
        xz_scale: f64,
        y_scale: f64,
        xz_factor: f64,
        y_factor: f64,
        smear_scale_multiplier: f64,
    ) -> Self {
        Self {
            min_limit_noise: PerlinNoise::create_for_blended_noise(
                rng,
                &(-15..=0).collect::<Vec<_>>(),
            ),
            max_limit_noise: PerlinNoise::create_for_blended_noise(
                rng,
                &(-15..=0).collect::<Vec<_>>(),
            ),
            main_noise: PerlinNoise::create_for_blended_noise(rng, &(-7..=0).collect::<Vec<_>>()),
            xz_scale,
            y_scale,
            xz_factor,
            y_factor,
            xz_multiplier: 684.412 * xz_scale,
            y_multiplier: 684.412 * y_scale,
            smear_scale_multiplier,
        }
    }
}

impl Noise3d for BlendedNoise {
    fn get(&self, pos: glam::DVec3) -> f64 {
        let multiplied_pos =
            pos * DVec3::new(self.xz_multiplier, self.y_multiplier, self.xz_multiplier);
        let scaled_pos = multiplied_pos / DVec3::new(self.xz_factor, self.y_factor, self.xz_factor);
        let smeared_y = self.y_multiplier * self.smear_scale_multiplier;
        let smeared_factored_y = smeared_y * self.y_factor;

        let mut main_res = 0.0;
        let mut current_factor = 1.0;
        for i in 0..8 {
            let noise = self.main_noise.get_octave_noise(i);
            if let Some(noise) = noise {
                main_res += noise.get_smeared(
                    scaled_pos.map(|v| PerlinNoise::wrap(v * current_factor)),
                    smeared_factored_y * current_factor,
                    scaled_pos.y * current_factor,
                ) / current_factor;
            }

            current_factor /= 2.0;
        }

        main_res = (main_res / 10.0 + 1.0) / 2.0;
        let min_limit = main_res < 1.0;
        let max_limit = main_res > 0.0;

        let mut current_factor = 1.0;
        let mut min_limit_res = 0.0;
        let mut max_limit_res = 0.0;
        for i in 0..16 {
            let wrapped_pos = multiplied_pos.map(|v| PerlinNoise::wrap(v * current_factor));
            if min_limit {
                let noise = self.min_limit_noise.get_octave_noise(i);
                if let Some(noise) = noise {
                    min_limit_res += noise.get_smeared(
                        wrapped_pos,
                        smeared_y * current_factor,
                        multiplied_pos.y * current_factor,
                    ) / current_factor;
                }
            }
            if max_limit {
                let noise = self.max_limit_noise.get_octave_noise(i);
                if let Some(noise) = noise {
                    max_limit_res += noise.get_smeared(
                        wrapped_pos,
                        smeared_y * current_factor,
                        multiplied_pos.y * current_factor,
                    ) / current_factor;
                }
            }
            current_factor /= 2.0;
        }

        clamped_lerp(min_limit_res / 512.0, max_limit_res / 512.0, main_res) / 128.0
    }
}

impl FunctionOp for BlendedNoise {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        self.get(pos.as_dvec3())
    }
}

pub struct Noise {
    noise: NormalNoise,
    xz_scale: f64,
    y_scale: f64,
}

impl Noise {
    pub fn new(rng: &impl Rng, parameters: NoiseParameters, xz_scale: f64, y_scale: f64) -> Self {
        Self {
            noise: NormalNoise::new(rng, parameters),
            xz_scale,
            y_scale,
        }
    }
}

impl FunctionOp for Noise {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        self.noise
            .get(pos.as_dvec3() * DVec3::new(self.xz_scale, self.y_scale, self.xz_scale))
    }
}

pub struct EndIslands(SimplexNoise);

// only do this once, because holy this is expensive
static END_ISLANDS_RNG: LazyLock<LegacyRng> = LazyLock::new(|| {
    let rng = LegacyRng::new(0);
    rng.skip_i32(17292);
    rng
});

impl EndIslands {
    pub fn new() -> Self {
        Self(SimplexNoise::new(&END_ISLANDS_RNG.clone()))
    }

    fn get_height_value(&self, pos: IVec2) -> f32 {
        let pos_d2 = pos / 2;
        let pos_m2 = pos % 2;
        let dist = 100.0 - (pos_d2.length_squared() as f32).sqrt() * 8.0;
        let mut res = dist.clamp(-100.0, 80.0);

        for x in -12..12 {
            for y in -12..12 {
                let relative_pos = IVec2::new(x, y);
                let current_pos = (pos_d2 + relative_pos).as_i64vec2();
                if current_pos.length_squared() > 4096
                    && self.0.get_2d(current_pos.as_dvec2()) < -0.9
                {
                    let factor = ((current_pos.x as f32).abs() * 3439.0
                        + (current_pos.y as f32) * 147.0)
                        % 13.0
                        + 9.0;
                    let pos2 = (pos_m2 - relative_pos * 2).as_vec2();
                    let sub_res = 100.0 - pos2.length() * factor;
                    let sub_res = sub_res.clamp(-100.0, 80.0);
                    res = res.max(sub_res);
                }
            }
        }

        res
    }
}

impl FunctionOp for EndIslands {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        (self.get_height_value((pos.xz() / 8).as_ivec2()) as f64 - 8.0) / 128.0
    }
}

impl Default for EndIslands {
    fn default() -> Self {
        Self::new()
    }
}

pub enum RarityValueMapper {
    Type1,
    Type2,
}

impl RarityValueMapper {
    fn map(&self, input: f64) -> f64 {
        match self {
            RarityValueMapper::Type1 => {
                if input < -0.5 {
                    0.75
                } else if input < 0.0 {
                    1.0
                } else if input < 0.5 {
                    1.5
                } else {
                    2.0
                }
            }
            RarityValueMapper::Type2 => {
                if input < -0.75 {
                    0.5
                } else if input < -0.5 {
                    0.75
                } else if input < 0.5 {
                    1.0
                } else if input < 0.75 {
                    2.0
                } else {
                    3.0
                }
            }
        }
    }
}

pub struct WeirdScalerSampler {
    rarity_value_mapper: RarityValueMapper,
    noise: NormalNoise,
    input: Box<dyn FunctionOp>,
}

impl WeirdScalerSampler {
    pub fn new(
        rng: &impl Rng,
        rarity_value_mapper: RarityValueMapper,
        parameters: NoiseParameters,
        input: Box<dyn FunctionOp>,
    ) -> Self {
        Self {
            rarity_value_mapper,
            noise: NormalNoise::new(rng, parameters),
            input,
        }
    }
}

impl FunctionOp for WeirdScalerSampler {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        let val = self.rarity_value_mapper.map(self.input.run_once(pos));
        val * self.noise.get(pos.as_dvec3() / val).abs()
    }
}

pub struct ShiftedNoise {
    noise: NormalNoise,
    xz_scale: f64,
    y_scale: f64,
    shift_x: Box<dyn FunctionOp>,
    shift_y: Box<dyn FunctionOp>,
    shift_z: Box<dyn FunctionOp>,
}

impl ShiftedNoise {
    pub fn new(
        rng: &impl Rng,
        parameters: NoiseParameters,
        xz_scale: f64,
        y_scale: f64,
        shift_x: Box<dyn FunctionOp>,
        shift_y: Box<dyn FunctionOp>,
        shift_z: Box<dyn FunctionOp>,
    ) -> Self {
        Self {
            noise: NormalNoise::new(rng, parameters),
            xz_scale,
            y_scale,
            shift_x,
            shift_y,
            shift_z,
        }
    }
}

impl FunctionOp for ShiftedNoise {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        self.noise.get(
            pos.as_dvec3() * DVec3::new(self.xz_scale, self.y_scale, self.xz_scale)
                + DVec3::new(
                    self.shift_x.run_once(pos),
                    self.shift_y.run_once(pos),
                    self.shift_z.run_once(pos),
                ),
        )
    }
}

pub struct RangeChoice {
    input: Box<dyn FunctionOp>,
    min_inclusive: f64,
    max_exclusive: f64,
    when_in_range: Box<dyn FunctionOp>,
    when_out_of_range: Box<dyn FunctionOp>,
}

impl FunctionOp for RangeChoice {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        let input = self.input.run_once(pos);
        if (self.min_inclusive..self.max_exclusive).contains(&input) {
            self.when_in_range.run_once(pos)
        } else {
            self.when_out_of_range.run_once(pos)
        }
    }
}

pub struct ShiftA(NormalNoise);

impl ShiftA {
    pub fn new(rng: &impl Rng, parameters: NoiseParameters) -> Self {
        Self(NormalNoise::new(rng, parameters))
    }
}

impl FunctionOp for ShiftA {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        self.0.get(pos.as_dvec3().with_y(0.0) / 4.0) * 4.0
    }
}

pub struct ShiftB(NormalNoise);

impl ShiftB {
    pub fn new(rng: &impl Rng, parameters: NoiseParameters) -> Self {
        Self(NormalNoise::new(rng, parameters))
    }
}

impl FunctionOp for ShiftB {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        self.0.get(pos.as_dvec3().with_z(0.0) / 4.0) * 4.0
    }
}

pub struct Shift(NormalNoise);

impl Shift {
    pub fn new(rng: &impl Rng, parameters: NoiseParameters) -> Self {
        Self(NormalNoise::new(rng, parameters))
    }
}

impl FunctionOp for Shift {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        self.0.get(pos.as_dvec3() / 4.0) * 4.0
    }
}

pub struct Clamp {
    pub input: Box<dyn FunctionOp>,
    pub min: f64,
    pub max: f64,
}

impl FunctionOp for Clamp {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        self.input.run_once(pos).clamp(self.min, self.max)
    }
}

pub struct Spline {
    pub coordinate: Box<dyn FunctionOp>,
    pub spline: spline::Spline,
}

impl FunctionOp for Spline {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        self.spline.compute(self.coordinate.run_once(pos) as f32) as f64
    }
}

pub struct Constant(pub f64);

impl FunctionOp for Constant {
    fn run_once(&self, _pos: &I64Vec3) -> f64 {
        self.0
    }
}

pub struct YClampedGradient {
    pub from_y: i32,
    pub to_y: i32,
    pub from_value: f64,
    pub to_value: f64,
}

impl FunctionOp for YClampedGradient {
    fn run_once(&self, pos: &I64Vec3) -> f64 {
        let t = (pos.y as f64 - self.from_y as f64) / (self.to_y - self.from_y) as f64;
        clamped_lerp(self.from_value, self.to_value, t)
    }

    fn run_plane(&self, pos: &I64Vec3) -> [f64; 16 * 16] {
        [self.run_once(pos); 16 * 16]
    }

    fn run_subchunk(&self, pos: &I64Vec3) -> [f64; SUBCHUNK_SIZE] {
        let mut res = [f64::NAN; SUBCHUNK_SIZE];
        for y in 0..16 {
            res[(y * 16 * 16)..((y + 1) * 16 * 16)]
                .fill(self.run_once(&pos.with_y(pos.y + y as i64)));
        }
        res
    }
}

macro_rules! unit_op {
    ($name:ident, $res_loc:literal) => {
        pub struct $name;

        impl FunctionOp for $name {
            fn run_once(&self, _pos: &I64Vec3) -> f64 {
                0.0
            }

            fn run_plane(&self, _pos: &I64Vec3) -> [f64; 16 * 16] {
                [0.0; 16 * 16]
            }

            fn run_subchunk(&self, _pos: &I64Vec3) -> [f64; SUBCHUNK_SIZE] {
                [0.0; SUBCHUNK_SIZE]
            }
        }

        impl NonDynFunctionOp for $name {
            const NAME: ResourceLocation<FunctionOpType> =
                ResourceLocation::new_static_mc($res_loc);
        }
    };
}

macro_rules! marker_op {
    ($name:ident, $res_loc:literal) => {
        pub struct $name(pub Box<dyn FunctionOp>);

        impl FunctionOp for $name {
            fn run_once(&self, pos: &I64Vec3) -> f64 {
                self.0.run_once(pos)
            }

            fn run_plane(&self, pos: &I64Vec3) -> [f64; 16 * 16] {
                self.0.run_plane(pos)
            }

            fn run_subchunk(&self, pos: &I64Vec3) -> [f64; SUBCHUNK_SIZE] {
                self.0.run_subchunk(pos)
            }
        }

        impl NonDynFunctionOp for $name {
            const NAME: ResourceLocation<FunctionOpType> =
                ResourceLocation::new_static_mc($res_loc);
        }
    };
}

macro_rules! monadic_op {
    ($name:ident, $res_loc:literal, $op:expr) => {
        pub struct $name(pub Box<dyn FunctionOp>);

        impl FunctionOp for $name {
            fn run_once(&self, pos: &I64Vec3) -> f64 {
                $op(self.0.run_once(pos))
            }

            fn run_plane(&self, pos: &I64Vec3) -> [f64; 16 * 16] {
                self.0.run_plane(pos).map($op)
            }

            fn run_subchunk(&self, pos: &I64Vec3) -> [f64; SUBCHUNK_SIZE] {
                self.0.run_subchunk(pos).map($op)
            }
        }

        impl NonDynFunctionOp for $name {
            const NAME: ResourceLocation<FunctionOpType> =
                ResourceLocation::new_static_mc($res_loc);
        }
    };
}

macro_rules! diadic_op {
    ($name:ident, $name_const:ident, $res_loc:literal, $op:ident) => {
        pub struct $name(pub Box<dyn FunctionOp>, pub Box<dyn FunctionOp>);

        impl FunctionOp for $name {
            fn run_once(&self, pos: &I64Vec3) -> f64 {
                self.0.run_once(pos).$op(self.1.run_once(pos))
            }

            fn run_plane(&self, pos: &I64Vec3) -> [f64; 16 * 16] {
                let mut res = [f64::NAN; 16 * 16];
                let in0 = self.0.run_plane(pos);
                let in1 = self.1.run_plane(pos);
                for x in 0..16 {
                    for z in 0..16 {
                        res[x + (z * 16)] = in0[x + (z * 16)].$op(in1[x + (z * 16)]);
                    }
                }
                res
            }

            fn run_subchunk(&self, pos: &I64Vec3) -> [f64; SUBCHUNK_SIZE] {
                let mut res = [f64::NAN; SUBCHUNK_SIZE];
                let in0 = self.0.run_subchunk(pos);
                let in1 = self.1.run_subchunk(pos);
                for x in 0..16 {
                    for z in 0..16 {
                        for y in 0..16 {
                            res[x + (z * 16) + (y * 16 * 16)] = in0[x + (z * 16) + (y * 16 * 16)]
                                .$op(in1[x + (z * 16) + (y * 16 * 16)]);
                        }
                    }
                }
                res
            }
        }

        impl NonDynFunctionOp for $name {
            const NAME: ResourceLocation<FunctionOpType> =
                ResourceLocation::new_static_mc($res_loc);
        }

        pub struct $name_const(pub Box<dyn FunctionOp>, pub f64);

        impl FunctionOp for $name_const {
            fn run_once(&self, pos: &I64Vec3) -> f64 {
                self.0.run_once(pos).$op(self.1)
            }

            fn run_plane(&self, pos: &I64Vec3) -> [f64; 16 * 16] {
                self.0.run_plane(pos).map(|v| v.$op(self.1))
            }

            fn run_subchunk(&self, pos: &I64Vec3) -> [f64; SUBCHUNK_SIZE] {
                self.0.run_subchunk(pos).map(|v| v.$op(self.1))
            }
        }
    };
}

// == UNIMPLEMENTED MARKER FUNCTIONS ==
marker_op! {CacheOnce, "cache_once"}
marker_op! {CacheAllInCell, "cache_all_in_cell"}

// == MAPPED FUNCTIONS ==
monadic_op! {Abs, "abs", |v: f64| v.abs()}
monadic_op! {Square, "square", |v: f64| v.powi(2)}
monadic_op! {Cube, "cube", |v: f64| v.powi(3)}
monadic_op! {HalfNegative, "half_negative", |v: f64| if v < 0.0 {v / 2.0} else {v}}
monadic_op! {QuarterNegative, "quarter_negative", |v: f64| if v < 0.0 {v / 4.0} else {v}}
monadic_op! {Squeeze, "squeeze", |v: f64| (v / 2.0) * (v * v * v / 24.0)}

// == FNS WITH TWO ARGUMENTS ==
diadic_op! {Add, AddConst, "add", add}
diadic_op! {Mul, MulConst, "mul", mul}
diadic_op! {Min, MinConst, "min", min}
diadic_op! {Max, MaxConst, "max", max}

// == OTHER UNIMPLEMENTED ==
unit_op! {BlendAlpha, "blend_alpha"}
unit_op! {BlendOffset, "blend_offset"}
marker_op! {BlendDensity, "blend_density"}
unit_op! {Beardifier, "beardifier"}

#[cfg(all(test, feature = "java_tests_module"))]
mod java_tests {
    use glam::DVec3;
    use jni::objects::JValue;

    use crate::{
        density_function::std_ops::BlendedNoise,
        java_tests::{self, get_jvm_env},
        noise::Noise3d as _,
    };

    #[test]
    fn blended_noise() {
        let mut env = get_jvm_env();
        let java_blended_noise = env
            .call_static(
                &java_tests::BlendedNoise::CREATE_UNSEEDED,
                &[
                    JValue::Double(1.0),
                    JValue::Double(1.0),
                    JValue::Double(1.0),
                    JValue::Double(1.0),
                    JValue::Double(1.0),
                ],
            )
            .l()
            .unwrap();
        let blended_noise = BlendedNoise::create_unseeded(1.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(
            env.call(
                &java_blended_noise,
                &java_tests::BlendedNoise::COMPUTE,
                &[
                    JValue::Double(0.0),
                    JValue::Double(0.0),
                    JValue::Double(0.0),
                ]
            )
            .d()
            .unwrap(),
            blended_noise.get(DVec3::new(0.0, 0.0, 0.0))
        );
        for x in -5..5 {
            for y in -5..5 {
                for z in -5..5 {
                    let x = x as f64 / 2.5;
                    let y = y as f64 / 2.5;
                    let z = z as f64 / 2.5;
                    assert_eq!(
                        env.call(
                            &java_blended_noise,
                            &java_tests::BlendedNoise::COMPUTE,
                            &[JValue::Double(x), JValue::Double(y), JValue::Double(z),]
                        )
                        .d()
                        .unwrap(),
                        blended_noise.get(DVec3::new(x, y, z)),
                        "at {x}, {y}, {z}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::I64Vec3;
    use mcpackloader::worldgen::NoiseParameters;

    use crate::{
        density_function::{
            FunctionOp,
            std_ops::{BlendedNoise, FlatCache, Interpolated},
        },
        noise::{Noise3d, NormalNoise},
        random::LegacyRng,
    };

    pub struct TestNoise(NormalNoise);

    impl TestNoise {
        pub fn new_boxed() -> Box<dyn FunctionOp> {
            Box::new(TestNoise(NormalNoise::new(
                &LegacyRng::new(0),
                NoiseParameters {
                    amplitudes: vec![1.0],
                    first_octave: -4,
                },
            )))
        }
    }

    impl FunctionOp for TestNoise {
        fn run_once(&self, pos: &I64Vec3) -> f64 {
            self.0.get(pos.as_dvec3())
        }

        fn run_plane(&self, pos: &I64Vec3) -> [f64; 16 * 16] {
            let mut res = [f64::NAN; 16 * 16];
            for x in 0..16 {
                for z in 0..16 {
                    res[x + (z * 16)] = self
                        .0
                        .get((pos + I64Vec3::new(x as i64, 0, z as i64)).as_dvec3());
                }
            }
            res
        }

        fn run_subchunk(&self, pos: &I64Vec3) -> [f64; crate::density_function::SUBCHUNK_SIZE] {
            let mut res = [f64::NAN; 16 * 16 * 16];
            for x in 0..16 {
                for z in 0..16 {
                    for y in 0..16 {
                        res[x + (z * 16) + (y * 16 * 16)] = self
                            .0
                            .get((pos + I64Vec3::new(x as i64, y as i64, z as i64)).as_dvec3());
                    }
                }
            }
            res
        }
    }

    #[test]
    fn test_noise() {
        function_op_tests(TestNoise::new_boxed().as_ref());
    }

    #[test]
    fn interpolated() {
        function_op_tests(&Interpolated(TestNoise::new_boxed()));
    }

    #[test]
    fn flat_cache() {
        function_op_tests(&FlatCache(TestNoise::new_boxed()));
    }

    #[test]
    fn blended_noise() {
        function_op_tests(&BlendedNoise::create_unseeded(1.0, 1.0, 1.0, 1.0, 1.0));
    }

    fn function_op_tests(op: &dyn FunctionOp) {
        let mut plane = op.run_plane(&I64Vec3::new(0, 0, 0));
        for x in 0..16 {
            for z in 0..16 {
                assert_more_or_less_eq!(
                    plane[x + (z * 16)],
                    op.run_once(&I64Vec3::new(x as i64, 0, z as i64)),
                    "at {x}, {z}"
                );
            }
        }
        plane = op.run_plane(&I64Vec3::new(16, 3, 16));
        for x in 0..16 {
            for z in 0..16 {
                assert_more_or_less_eq!(
                    plane[x + (z * 16)],
                    op.run_once(&I64Vec3::new(x as i64 + 16, 3, z as i64 + 16)),
                    "at {x}, {z}"
                );
            }
        }
        plane = op.run_plane(&I64Vec3::new(-16, -3, -16));
        for x in 0..16 {
            for z in 0..16 {
                assert_more_or_less_eq!(
                    plane[x + (z * 16)],
                    op.run_once(&I64Vec3::new(x as i64 - 16, -3, z as i64 - 16)),
                    "at {x}, {z}"
                );
            }
        }
        let mut subchunk = op.run_subchunk(&I64Vec3::new(0, 0, 0));
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..16 {
                    assert_more_or_less_eq!(
                        subchunk[x + (z * 16) + (y * 16 * 16)],
                        op.run_once(&I64Vec3::new(x as i64, y as i64, z as i64)),
                        "at {x}, {y}, {z}"
                    );
                }
            }
        }
        subchunk = op.run_subchunk(&I64Vec3::new(16, 16, 16));
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..16 {
                    assert_more_or_less_eq!(
                        subchunk[x + (z * 16) + (y * 16 * 16)],
                        op.run_once(&I64Vec3::new(x as i64 + 16, y as i64 + 16, z as i64 + 16)),
                        "at {x}, {y}, {z}"
                    );
                }
            }
        }
        subchunk = op.run_subchunk(&I64Vec3::new(-16, -16, -16));
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..16 {
                    assert_more_or_less_eq!(
                        subchunk[x + (z * 16) + (y * 16 * 16)],
                        op.run_once(&I64Vec3::new(x as i64 - 16, y as i64 - 16, z as i64 - 16)),
                        "at {x}, {y}, {z}"
                    );
                }
            }
        }
    }
}
