/// Macro for asserting that two floating point values are equal, allowing for some rounding errors
#[allow(unused_macros, reason = "its used, idk why it won't say that tho")]
macro_rules! assert_more_or_less_eq {
    ($left:expr, $right:expr) => {
        let left_val = $left;
        let right_val = $right;
        if (left_val - right_val).abs() > 1e-10 {
            // this will always fail, only used to create similar error messages
            assert_eq!(left_val, right_val);
        }
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        let left_val = $left;
        let right_val = $right;
        if (left_val - right_val).abs() > 1e-10 {
            // this will always fail, only used to create similar error messages
            assert_eq!(left_val, right_val, $($arg)+);
        }
    };
}

pub mod density_function;
mod helpers;
#[cfg(all(test, feature = "java_tests_module"))]
mod java_tests;
pub mod noise;
pub mod random;
pub mod registry;
mod spline;
