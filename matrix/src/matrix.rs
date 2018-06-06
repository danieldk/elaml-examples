//! Module for matrices.

#[cfg(feature = "simd-accel")]
use simd::f32x4;

#[cfg(feature = "avx-accel")]
use simd::x86::avx::f32x8;

/// Matrix storage orders.
#[derive(Copy, Clone, Debug)]
pub enum Order {
    /// Column-major order.
    ColumnMajor,

    /// Row-major order.
    RowMajor,
}

/// A matrix data type, represents two-dimensional tensors.
pub struct Matrix {
    order: Order,
    data: Vec<f32>,
    n_rows: usize,
    n_cols: usize,
}

impl Matrix {
    pub fn from_vec(order: Order, data: Vec<f32>, n_rows: usize, n_cols: usize) -> Self {
        assert!(n_rows > 0);
        assert!(n_cols > 0);
        assert_eq!(data.len(), n_rows * n_cols);

        Matrix {
            order,
            data,
            n_rows,
            n_cols,
        }
    }

    /// Construct a new matrix with the given order, number of rows,
    /// and columns. The matrix will be zero-initialized.
    pub fn zeros(order: Order, n_rows: usize, n_cols: usize) -> Self {
        assert!(n_rows > 0);
        assert!(n_cols > 0);

        Matrix {
            order,
            data: vec![0f32; n_rows * n_cols],
            n_rows,
            n_cols,
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn frobenius_norm(&self) -> f32 {
        l2_norm(&self.data)
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[self.index(row, col)]
    }

    fn index(&self, row: usize, col: usize) -> usize {
        assert!(row < self.n_rows, "Row {} >= bound {}", row, self.n_rows);
        assert!(col < self.n_cols, "Column {} >= bound {}", col, self.n_cols);

        match self.order {
            Order::RowMajor => row * self.n_cols + col,
            Order::ColumnMajor => col * self.n_rows + row,
        }
    }

    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.n_cols, other.n_rows);

        let mut c = Matrix::zeros(Order::RowMajor, self.n_rows, other.n_cols);

        for i in 0..self.n_rows {
            for j in 0..other.n_cols {
                let mut sum = 0f32;

                for k in 0..self.n_cols {
                    sum += self.get(i, k) * other.get(k, j);
                }

                c.set(i, j, sum);
            }
        }

        c
    }

    pub fn order(&self) -> Order {
        self.order
    }

    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        let idx = self.index(row, col);
        self.data[idx] = val;
    }
}

cfg_if! {
    if #[cfg(feature = "avx-accel")] {
        fn l2_norm(s: &[f32]) -> f32 {
            l2_norm_f32x8(s)
        }
    } else if #[cfg(feature = "simd-accel")] {
        fn l2_norm(s: &[f32]) -> f32 {
            l2_norm_f32x4(s)
        }
    } else {
        fn l2_norm(s: &[f32]) -> f32 {
            l2_norm_unvectorized(s)
        }
    }
}

#[allow(dead_code)]
pub(crate) fn l2_norm_unvectorized(s: &[f32]) -> f32 {
    s.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

#[allow(dead_code)]
#[cfg(feature = "simd-accel")]
fn l2_norm_f32x4(mut s: &[f32]) -> f32 {
    let mut sums = f32x4::splat(0.0);

    while s.len() >= 4 {
        let data = f32x4::load(s, 0);
        sums = sums + data * data;
        s = &s[4..];
    }

    let sum = sums.extract(0) + sums.extract(1) + sums.extract(2) + sums.extract(3);

    let rest_sum = s.iter().map(|&v| v * v).sum::<f32>();

    (sum + rest_sum).sqrt()
}

#[cfg(feature = "avx-accel")]
fn l2_norm_f32x8(mut s: &[f32]) -> f32 {
    let mut sums = f32x8::splat(0.0);

    while s.len() >= 8 {
        let data = f32x8::load(s, 0);
        sums = sums + data * data;
        s = &s[8..];
    }

    let sum = sums.extract(0)
        + sums.extract(1)
        + sums.extract(2)
        + sums.extract(3)
        + sums.extract(4)
        + sums.extract(5)
        + sums.extract(6)
        + sums.extract(7);

    let rest_sum = s.iter().map(|&v| v * v).sum::<f32>();

    (sum + rest_sum).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{l2_norm_unvectorized, Matrix, Order};

    fn close(a: f32, b: f32, eps: f32) -> bool {
        let diff = (a - b).abs();
        if diff > eps {
            return false;
        }

        true
    }

    fn all_close(a: &[f32], b: &[f32], eps: f32) -> bool {
        for (&av, &bv) in a.iter().zip(b) {
            if !close(av, bv, eps) {
                return false;
            }
        }

        true
    }

    #[test]
    pub fn l2_norm_unvectorized_test() {
        let v = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0];
        assert!(close(l2_norm_unvectorized(&v), 16.881943, 1e-6));
    }

    quickcheck!{
        fn frobenius_norm_test(u: Vec<f32>) -> bool {
            if u.is_empty() {
                return true
            }

            let len = u.len();
            let m = Matrix::from_vec(Order::ColumnMajor, u, 1, len);
            close(m.frobenius_norm(), l2_norm_unvectorized(m.as_slice()), 1e-2)
        }
    }

    #[test]
    pub fn get_order_test() {
        let rm = Matrix::from_vec(Order::RowMajor, (1..=6).map(|v| v as f32).collect(), 3, 2);
        assert!(close(rm.get(0, 0), 1.0, 0.0));
        assert!(close(rm.get(0, 1), 2.0, 0.0));
        assert!(close(rm.get(1, 0), 3.0, 0.0));
        assert!(close(rm.get(1, 1), 4.0, 0.0));
        assert!(close(rm.get(2, 0), 5.0, 0.0));
        assert!(close(rm.get(2, 1), 6.0, 0.0));

        let cm = Matrix::from_vec(
            Order::ColumnMajor,
            (1..=6).map(|v| v as f32).collect(),
            3,
            2,
        );
        assert!(close(cm.get(0, 0), 1.0, 0.0));
        assert!(close(cm.get(0, 1), 4.0, 0.0));
        assert!(close(cm.get(1, 0), 2.0, 0.0));
        assert!(close(cm.get(1, 1), 5.0, 0.0));
        assert!(close(cm.get(2, 0), 3.0, 0.0));
        assert!(close(cm.get(2, 1), 6.0, 0.0));
    }

    #[test]
    pub fn matmul_test() {
        let a = Matrix::from_vec(Order::RowMajor, (1..=9).map(|v| v as f32).collect(), 3, 3);
        let b = Matrix::from_vec(Order::RowMajor, (1..=6).map(|v| v as f32).collect(), 3, 2);

        let result = a.matmul(&b);

        assert!(all_close(
            result.as_slice(),
            &[22.0, 28.0, 49.0, 64.0, 76.0, 100.0],
            1e-6
        ));
    }

    #[test]
    pub fn set_order_test() {
        let mut rm = Matrix::zeros(Order::RowMajor, 3, 2);
        rm.set(0, 0, 1.0);
        rm.set(0, 1, 2.0);
        rm.set(1, 0, 3.0);
        rm.set(1, 1, 4.0);
        rm.set(2, 0, 5.0);
        rm.set(2, 1, 6.0);
        assert!(all_close(
            rm.as_slice(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            1e-6
        ));

        let mut cm = Matrix::zeros(Order::ColumnMajor, 3, 2);
        cm.set(0, 0, 1.0);
        cm.set(0, 1, 2.0);
        cm.set(1, 0, 3.0);
        cm.set(1, 1, 4.0);
        cm.set(2, 0, 5.0);
        cm.set(2, 1, 6.0);
        assert!(all_close(
            cm.as_slice(),
            &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
            1e-6
        ));
    }
}
