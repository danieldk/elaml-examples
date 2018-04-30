#![feature(test)]

extern crate basic_types;
extern crate rand;
extern crate test;

use std::iter;

use basic_types::popcnt::{popcnt, popcnt_lookup};
use rand::{weak_rng, Rng};
use test::{black_box, Bencher};

fn random_vec<R>(rng: &mut R, len: usize) -> Vec<u64>
where
    R: Rng,
{
    iter::repeat(()).map(|()| rng.gen()).take(len).collect()
}

#[bench]
fn popcnt_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let v = black_box(random_vec(&mut rng, 1000));

    b.iter(move || {
        for val in v.iter() {
            black_box(popcnt(*val));
        }
    })
}

#[bench]
fn popcnt_table_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let v = black_box(random_vec(&mut rng, 1000));

    b.iter(move || {
        for val in v.iter() {
            black_box(popcnt_lookup(*val));
        }
    })
}
