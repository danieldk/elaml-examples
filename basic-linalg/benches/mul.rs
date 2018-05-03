#![feature(test)]

extern crate basic_linalg;
extern crate rand;
extern crate test;

use std::iter;

use basic_linalg::mul::{mul, mul_inplace, mul_slow};
use rand::{weak_rng, Rng};
use test::{black_box, Bencher};

fn random_vec<R>(rng: &mut R, len: usize) -> Vec<f32>
where
    R: Rng,
{
    iter::repeat(()).map(|()| rng.gen()).take(len).collect()
}

#[bench]
fn mul_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let v1 = black_box(random_vec(&mut rng, 500));
    let v2 = black_box(random_vec(&mut rng, 500));

    b.iter(|| {
        // Add copy overhead for fair comparison. See mul_inplace_bench
        // for more information.
        let v1 = black_box(v1.clone());
        black_box(mul(&v1, &v2));
    })
}

#[bench]
fn mul_inplace_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let v1 = black_box(random_vec(&mut rng, 500));
    let v2 = black_box(random_vec(&mut rng, 500));

    b.iter(|| {
        // In the clase of in-place multiplication, v1 is modified, marking
        // the cache pages dirty. Consequently, the corresponding memory in
        // RAM has to be updated. This makes benchmarking of the in-place
        // multiplication unfair, since it is also benchmarking these
        // flushes. So, instead work on a copy of v1 that gets discarded to
        // make the result comparable to mul_bench.
        let mut v1 = black_box(v1.clone());
        black_box(mul_inplace(&mut v1, &v2));
    })
}

#[bench]
fn mul_slow_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let v1 = black_box(random_vec(&mut rng, 500));
    let v2 = black_box(random_vec(&mut rng, 500));

    b.iter(|| {
        // Add copy overhead for fair comparison. See mul_inplace_bench
        // for more information.
        let v1 = black_box(v1.clone());
        black_box(mul_slow(&v1, &v2));
    })
}
