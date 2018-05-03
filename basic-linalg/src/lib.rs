#![feature(cfg_target_feature)]

#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

#[cfg(test)]
extern crate rand;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

extern crate simd;

pub mod dot;

pub mod dot_simd;

pub mod mul;
