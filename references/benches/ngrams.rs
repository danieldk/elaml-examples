#![feature(test)]

extern crate rand;
extern crate references;

extern crate test;

use std::iter;

use rand::{weak_rng, Rand, Rng};
use test::{black_box, Bencher};

use references::ngrams::ngrams;

fn random_vec<R, T>(rng: &mut R, len: usize) -> Vec<T>
where
    R: Rng,
    T: Rand,
{
    iter::repeat(()).map(|()| rng.gen()).take(len).collect()
}

fn random_vecs<R, T>(rng: &mut R, min_len: usize, max_len: usize, n: usize) -> Vec<Vec<T>>
where
    R: Rng,
    T: Rand,
{
    assert!(min_len > 0);
    assert!(max_len >= min_len);

    let mut vecs = Vec::new();

    for _ in 0..n {
        let len = rng.gen_range(min_len, max_len + 1);
        vecs.push(random_vec(rng, len));
    }

    vecs
}

#[bench]
fn ngrams_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let strings: Vec<Vec<char>> = black_box(random_vecs(&mut rng, 10, 12, 100));
    b.iter(|| {
        for string in &strings {
            black_box(ngrams(string, 1, 3));
        }
    })
}
