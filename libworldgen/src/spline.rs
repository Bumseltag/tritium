//! Minecrafts implementation of cubic splines, simplified and rewritten.
//!
//! Some things have been renamed:
//! - `CubicSpline` -> [`SplineValue`]
//! - `MultiPoint` -> [`Spline`]

use crate::helpers::lerp_f32;

pub enum SplineValue {
    Constant(f32),
    SubSpline(Spline),
}

impl SplineValue {
    pub fn compute(&self, coordinate_input: f32) -> f32 {
        match self {
            SplineValue::Constant(value) => *value,
            SplineValue::SubSpline(mp) => mp.compute(coordinate_input),
        }
    }
}

/// A point of a cubic spline.
pub struct Point {
    pub location: f32,
    pub value: SplineValue,
    pub derivative: f32,
}

/// A cubic spline.
///
/// A spline consists of some [`Point`]s that each have an x coordinate (`location`), y coordinate (`value`) and a derivative.
/// `value` can also be itself a Spline ([`SplineValue::SubSpline`]),
/// in which case it will get interpolated in some weird fashion.
pub struct Spline {
    pub points: Vec<Point>,
}

impl Spline {
    pub fn compute(&self, coordinate_input: f32) -> f32 {
        let i_usize = match self
            .points
            .binary_search_by(|p| p.location.partial_cmp(&coordinate_input).unwrap())
        {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        // Adjust 'i' to be the index of the point to the left of coordinate_input
        let i = if i_usize == 0 { 0 } else { i_usize - 1 };

        // Handle cases outside the defined points using linear extension
        let first = self.points.first().unwrap();
        if coordinate_input < first.location {
            return first.value.compute(first.location)
                + first.derivative * (coordinate_input - first.location);
        }
        let last = self.points.first().unwrap();
        if coordinate_input > last.location {
            return last.value.compute(last.location)
                + last.derivative * (coordinate_input - last.location);
        }

        // Standard cubic spline interpolation
        let loc0 = self.points[i].location;
        let loc1 = self.points[i + 1].location;
        let der0 = self.points[i].derivative;
        let der1 = self.points[i + 1].derivative;
        let f = (coordinate_input - loc0) / (loc1 - loc0);

        let val0 = self.points[i].value.compute(loc0);
        let val1 = self.points[i + 1].value.compute(loc1);

        let f8 = der0 * (loc1 - loc0) - (val1 - val0);
        let f9 = -der1 * (loc1 - loc0) + (val1 - val0);

        lerp_f32(val0, val1, f) + (f * (1.0 - f)) * lerp_f32(f8, f9, f)
    }
}

impl From<f32> for SplineValue {
    fn from(value: f32) -> Self {
        SplineValue::Constant(value)
    }
}
