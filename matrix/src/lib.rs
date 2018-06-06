#[macro_use]
extern crate cfg_if;

#[cfg(feature = "simd-accel")]
extern crate simd;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
extern crate rand;

pub mod matrix;
