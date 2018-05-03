#![feature(test)]

extern crate basic_linalg;
extern crate rand;
extern crate test;

use std::iter;

use basic_linalg::dot::{dot, dot_unrolled};
use basic_linalg::dot_simd::dot_f32x4;
#[cfg(target_feature = "avx")]
use basic_linalg::dot_simd::dot_f32x8;
use rand::{weak_rng, Rng};
use test::{black_box, Bencher};

fn random_vec<R>(rng: &mut R, len: usize) -> Vec<f32>
where
    R: Rng,
{
    iter::repeat(()).map(|()| rng.gen()).take(len).collect()
}

#[bench]
fn dot_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let v1 = black_box(random_vec(&mut rng, 500));
    let v2 = black_box(random_vec(&mut rng, 500));

    b.iter(|| {
        black_box(dot(&v1, &v2));
    })
}

#[bench]
fn dot_f32x4_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let v1 = black_box(random_vec(&mut rng, 500));
    let v2 = black_box(random_vec(&mut rng, 500));

    b.iter(|| {
        black_box(dot_f32x4(&v1, &v2));
    })
}

#[cfg(target_feature = "avx")]
#[bench]
fn dot_f32x8_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let v1 = black_box(random_vec(&mut rng, 500));
    let v2 = black_box(random_vec(&mut rng, 500));

    b.iter(|| {
        black_box(dot_f32x8(&v1, &v2));
    })
}

#[bench]
fn dot_unrolled_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let v1 = black_box(random_vec(&mut rng, 500));
    let v2 = black_box(random_vec(&mut rng, 500));

    b.iter(|| {
        black_box(dot_unrolled(&v1, &v2));
    })
}
