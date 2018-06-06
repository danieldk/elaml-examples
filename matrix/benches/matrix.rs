#![feature(test)]

extern crate matrix;
extern crate rand;
extern crate test;

use std::iter;

use matrix::matrix::{Matrix, Order};
use rand::{weak_rng, Rng};
use test::{black_box, Bencher};

// const M: usize = 1;
// const N: usize = 4000;
// const O: usize = 4000;

const M: usize = 200;
const N: usize = 200;
const O: usize = 200;

fn random_matrix<R>(rng: &mut R, order: Order, rows: usize, cols: usize) -> Matrix
where
    R: Rng,
{
    let n_cells = rows * cols;

    let data = iter::repeat(()).map(|()| rng.gen()).take(n_cells).collect();

    Matrix::from_vec(order, data, rows, cols)
}

#[bench]
pub fn frobenius_norm_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let m = black_box(random_matrix(&mut rng, Order::RowMajor, M, N));

    b.iter(|| {
        black_box(m.frobenius_norm());
    })
}

#[bench]
pub fn matmul_rm_rm_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let m = black_box(random_matrix(&mut rng, Order::RowMajor, M, N));
    let n = black_box(random_matrix(&mut rng, Order::RowMajor, N, O));

    b.iter(|| {
        black_box(m.matmul(&n));
    })
}

#[bench]
pub fn matmul_rm_cm_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let m = black_box(random_matrix(&mut rng, Order::RowMajor, M, N));
    let n = black_box(random_matrix(&mut rng, Order::ColumnMajor, N, O));

    b.iter(|| {
        black_box(m.matmul(&n));
    })
}

#[bench]
pub fn matmul_cm_rm_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let m = black_box(random_matrix(&mut rng, Order::ColumnMajor, M, N));
    let n = black_box(random_matrix(&mut rng, Order::RowMajor, N, O));

    b.iter(|| {
        black_box(m.matmul(&n));
    })
}

#[bench]
pub fn matmul_cm_cm_bench(b: &mut Bencher) {
    let mut rng = weak_rng();
    let m = black_box(random_matrix(&mut rng, Order::ColumnMajor, M, N));
    let n = black_box(random_matrix(&mut rng, Order::ColumnMajor, N, O));

    b.iter(|| {
        black_box(m.matmul(&n));
    })
}
