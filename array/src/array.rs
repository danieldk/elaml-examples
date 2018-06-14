//! Module for tensors.

use std::ops::Mul;

pub type Ix0 = [usize; 0];

pub type Ix1 = [usize; 1];

pub type Ix2 = [usize; 2];

pub type Ix3 = [usize; 3];

pub trait Dim: Copy {
    fn as_slice(&self) -> &[usize];

    /// The size of this shape.
    ///
    /// The size is the total number of elements.
    fn size(&self) -> usize {
        self.as_slice().iter().product()
    }

    fn valid_index(&self, idx: &Self) -> bool {
        for (shape, idx) in self.as_slice().iter().zip(idx.as_slice()) {
            if idx >= shape {
                return false;
            }
        }

        return true;
    }
}

impl Dim for Ix0 {
    fn as_slice(&self) -> &[usize] {
        self
    }
}

impl Dim for Ix1 {
    fn as_slice(&self) -> &[usize] {
        self
    }
}

impl Dim for Ix2 {
    fn as_slice(&self) -> &[usize] {
        self
    }
}

impl Dim for Ix3 {
    fn as_slice(&self) -> &[usize] {
        self
    }
}

/// A matrix data type, represents two-dimensional tensors.
pub struct Array<D> {
    data: Vec<f32>,
    shape: D,
}

impl<D> Array<D>
where
    D: Dim,
{
    pub fn from_vec(shape: D, data: Vec<f32>) -> Self {
        assert!(shape.as_slice().iter().all(|&v| v > 0));
        assert_eq!(
            data.len(),
            shape.size(),
            "Shape size {} does not correspond to Vec size {}",
            shape.size(),
            data.len()
        );

        Array { data, shape: shape }
    }

    /// Construct a new matrix with the given order, number of rows,
    /// and columns. The matrix will be zero-initialized.
    pub fn zeros(shape: D) -> Self {
        assert!(shape.as_slice().iter().all(|&v| v > 0));

        Array {
            data: vec![0f32; shape.size()],
            shape,
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn get(&self, index: D) -> f32 {
        self.data[self.index(index)]
    }

    fn index(&self, index: D) -> usize {
        assert!(self.shape.valid_index(&index));

        let index_s = index.as_slice();
        let shape_s = self.shape.as_slice();

        let mut idx = 0;
        for i in 0..index_s.len() {
            idx += index_s[i] * shape_s[i + 1..].iter().product::<usize>();
        }

        idx
    }

    pub fn set(&mut self, index: D, val: f32) {
        let idx = self.index(index);
        self.data[idx] = val;
    }
}

impl<'a, D> Mul for &'a Array<D>
where
    D: Dim,
{
    type Output = Array<D>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape.as_slice(),
            rhs.shape.as_slice(),
            "Shapes of multiplied tensors should be the same."
        );

        let output_data = self.data
            .iter()
            .zip(&rhs.data)
            .map(|(z, o)| z * o)
            .collect();
        Array::from_vec(self.shape, output_data)
    }
}

impl<D> Mul for Array<D>
where
    D: Dim,
{
    type Output = Array<D>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape.as_slice(),
            rhs.shape.as_slice(),
            "Shapes of multiplied tensors should be the same."
        );

        let output_data = self.data
            .into_iter()
            .zip(rhs.data)
            .map(|(z, o)| z * o)
            .collect();
        Array::from_vec(self.shape, output_data)
    }
}

pub trait Dot {
    type Output;

    fn dot(&self, rhs: &Self) -> Self::Output;
}

impl Dot for Array<Ix1> {
    type Output = Array<Ix0>;

    fn dot(&self, rhs: &Self) -> Self::Output {
        assert_eq!(self.shape[0], rhs.shape[0]);
        Array::from_vec(
            [],
            vec![
                self.data
                    .iter()
                    .zip(&rhs.data)
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>(),
            ],
        )
    }
}

impl Dot for Array<Ix2> {
    type Output = Array<Ix2>;

    fn dot(&self, rhs: &Self) -> Self::Output {
        assert_eq!(self.shape[1], rhs.shape[0]);

        let mut c = Array::zeros([self.shape[0], rhs.shape[1]]);

        for i in 0..self.shape[0] {
            for j in 0..rhs.shape[1] {
                let mut sum = 0f32;

                for k in 0..self.shape[1] {
                    sum += self.get([i, k]) * rhs.get([k, j]);
                }

                c.set([i, j], sum);
            }
        }

        c
    }
}

#[cfg(test)]
mod tests {
    use super::{Array, Dot};

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
    pub fn get_order0_test() {
        let a = Array::from_vec([], vec![42.0]);
        assert_eq!(a.get([]), 42.0);
    }

    #[test]
    pub fn get_order1_test() {
        let a = Array::from_vec([6], (1..=6).map(|v| v as f32).collect());
        assert_eq!(a.get([0]), 1.0);
        assert_eq!(a.get([1]), 2.0);
        assert_eq!(a.get([2]), 3.0);
        assert_eq!(a.get([3]), 4.0);
        assert_eq!(a.get([4]), 5.0);
        assert_eq!(a.get([5]), 6.0);
    }

    #[test]
    pub fn get_order2_test() {
        let rm = Array::from_vec([3, 2], (1..=6).map(|v| v as f32).collect());
        assert_eq!(rm.get([0, 0]), 1.0);
        assert_eq!(rm.get([0, 1]), 2.0);
        assert_eq!(rm.get([1, 0]), 3.0);
        assert_eq!(rm.get([1, 1]), 4.0);
        assert_eq!(rm.get([2, 0]), 5.0);
        assert_eq!(rm.get([2, 1]), 6.0)
    }

    #[test]
    pub fn get_order3_test() {
        let rm = Array::from_vec([2, 2, 2], (1..=8).map(|v| v as f32).collect());
        assert_eq!(rm.get([0, 0, 0]), 1.0);
        assert_eq!(rm.get([0, 0, 1]), 2.0);
        assert_eq!(rm.get([0, 1, 0]), 3.0);
        assert_eq!(rm.get([0, 1, 1]), 4.0);
        assert_eq!(rm.get([1, 0, 0]), 5.0);
        assert_eq!(rm.get([1, 0, 1]), 6.0);
        assert_eq!(rm.get([1, 1, 0]), 7.0);
        assert_eq!(rm.get([1, 1, 1]), 8.0);
    }

    #[test]
    pub fn mul_test() {
        let u = Array::from_vec([6], (1..=6).map(|v| v as f32).collect());
        let v = Array::from_vec([6], (1..=6).map(|v| v as f32).collect());
        let p = u * v;

        assert!(all_close(
            &vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
            &p.data,
            1e-6
        ));
    }

    #[test]
    pub fn dot_rank_1_test() {
        let a = Array::from_vec([5], (1..=5).map(|v| v as f32).collect());
        let b = Array::from_vec([5], (1..=5).map(|v| v as f32).collect());

        assert!(all_close(&[55.0], a.dot(&b).as_slice(), 1e-6));
    }

    #[test]
    pub fn dot_rank_2_test() {
        let a = Array::from_vec([3, 3], (1..=9).map(|v| v as f32).collect());
        let b = Array::from_vec([3, 2], (1..=6).map(|v| v as f32).collect());

        let result = a.dot(&b);

        assert!(all_close(
            result.as_slice(),
            &[22.0, 28.0, 49.0, 64.0, 76.0, 100.0],
            1e-6
        ));
    }

    #[test]
    pub fn set_order_test() {
        let mut rm = Array::zeros([3, 2]);
        rm.set([0, 0], 1.0);
        rm.set([0, 1], 2.0);
        rm.set([1, 0], 3.0);
        rm.set([1, 1], 4.0);
        rm.set([2, 0], 5.0);
        rm.set([2, 1], 6.0);
        assert!(all_close(
            rm.as_slice(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            1e-6
        ));
    }
}
