use std::error::Error;

use mcpackloader::{ResourceLocation, worldgen::NoiseParameters};

pub trait Registry {
    type Error: Error;
    fn get_noise_params(
        &mut self,
        res: &ResourceLocation<NoiseParameters>,
    ) -> Result<&NoiseParameters, &Self::Error>;
}
