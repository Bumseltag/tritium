use glam::I64Vec3;
use mcpackloader::ResourceLocation;

pub mod std_ops;

pub const SUBCHUNK_SIZE: usize = 16 * 16 * 16;

pub trait FunctionOp {
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

pub trait NonDynFunctionOp: FunctionOp {
    const NAME: ResourceLocation<FunctionOpType>;
}

pub struct FunctionOpType;
