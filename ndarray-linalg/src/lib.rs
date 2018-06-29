#[cfg(test)]
#[macro_use]
extern crate approx;
extern crate ndarray;
extern crate num_traits;
extern crate ordered_float;

use std::iter::Sum;

use ndarray::{Array1, ArrayBase, Data, Dimension, Ix1, Ix2};
use num_traits::Float;
use ordered_float::OrderedFloat;

pub trait EuclideanDistance<T> {
    type Output;

    fn euclidean_distance(&self, other: &T) -> Self::Output;
}

impl<A, S, S2> EuclideanDistance<ArrayBase<S2, Ix1>> for ArrayBase<S, Ix1>
where
    A: Float + Sum,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    type Output = A;

    fn euclidean_distance(&self, other: &ArrayBase<S2, Ix1>) -> Self::Output {
        assert_eq!(self.shape(), other.shape());
        
        let diff = self - other;
        diff.norm(Norms::L2)
    }
}

impl<A, S, S2> EuclideanDistance<ArrayBase<S2, Ix1>> for ArrayBase<S, Ix2>
where
    A: Float + Sum,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    type Output = Array1<A>;

    fn euclidean_distance(&self, other: &ArrayBase<S2, Ix1>) -> Self::Output {
        assert_eq!(self.shape()[1], other.shape()[0]);
        
        let diff = self - other;
        diff.outer_iter().map(|v| v.norm(Norms::L2)).collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Norms {
    L1,
    L2,
    Infinity,
}

pub trait Norm {
    type Output;

    fn norm(&self, norm: Norms) -> Self::Output;
}

impl<A, D, S> Norm for ArrayBase<S, D>
where
    A: Float + Sum,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = A;

    fn norm(&self, norm: Norms) -> Self::Output {
        use Norms::*;

        match norm {
            L1 => self.iter().map(|v| v.abs()).sum(),
            L2 => self.iter().map(|&v| v * v).sum::<A>().sqrt(),
            Infinity => self
                .iter()
                .map(|v| v.abs())
                .max_by(|&v1, &v2| OrderedFloat(v1).cmp(&OrderedFloat(v2)))
                .unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use super::{EuclideanDistance, Norm, Norms};

    #[test]
    fn l1_norm_test() {
        assert_eq!(arr1(&[0.0, -1.0, 2.0, -3.0]).norm(Norms::L1), 6.0);
    }

    #[test]
    fn l2_norm_test() {
        assert_abs_diff_eq!(
            arr1(&[0.0, -1.0, 2.0, -3.0]).norm(Norms::L2),
            3.741657,
            epsilon = 1e-6
        );
    }

    #[test]
    fn infinity_norm_test() {
        assert_eq!(arr1(&[1.0, -4.0, 3.0, 1.5]).norm(Norms::Infinity), 4.0);
    }

    #[test]
    fn euclidean_distance_test() {
        assert_abs_diff_eq!(
            arr1(&[1.0, 0.0, 2.0, -1.0]).euclidean_distance(&arr1(&[1.0, 1.0, 1.0, -3.0])),
            2.449490,
            epsilon = 1e-6
        );

        let dists = arr2(&[[1.0, 0.0, 2.0, -1.0], [-2.0, 3.0, 1.5, 6.0]])
            .euclidean_distance(&arr1(&[1.0, 1.0, 1.0, -3.0]));
        assert_abs_diff_eq!(dists[0], 2.449490, epsilon = 1e-6);
        assert_abs_diff_eq!(dists[1], 9.708244, epsilon = 1e-6);
    }
}
