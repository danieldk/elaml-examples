use simd::f32x4;
#[cfg(target_feature = "avx")]
use simd::x86::avx::f32x8;

use dot::dot;

pub fn dot_f32x4(mut u: &[f32], mut v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len());

    let mut sums = f32x4::splat(0.0);

    while u.len() >= 4 {
        let a = f32x4::load(u, 0);
        let b = f32x4::load(v, 0);

        sums = sums + a * b;

        u = &u[4..];
        v = &v[4..];
    }

    sums.extract(0) + sums.extract(1) + sums.extract(2) + sums.extract(3) + dot(u, v)
}

#[cfg(target_feature = "avx")]
pub fn dot_f32x8(mut u: &[f32], mut v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len());

    let mut sums = f32x8::splat(0.0);

    while u.len() >= 8 {
        let a = f32x8::load(u, 0);
        let b = f32x8::load(v, 0);

        sums = sums + a * b;

        u = &u[8..];
        v = &v[8..];
    }

    sums.extract(0) + sums.extract(1) + sums.extract(2) + sums.extract(3) + sums.extract(4)
        + sums.extract(5) + sums.extract(6) + sums.extract(7) + dot(u, v)
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use dot::dot;

    use super::dot_f32x4;
    #[cfg(target_feature = "avx")]
    use super::dot_f32x8;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-1
    }

    quickcheck!{
       fn dot_f32x4_test(u: Vec<f32>, v: Vec<f32>) -> bool {
            let len = cmp::min(u.len(), v.len());
            let u = &u[..len];
            let v = &v[..len];
            approx_eq(dot_f32x4(u, v), dot(u, v))

        }
    }

    #[cfg(target_feature = "avx")]
    quickcheck!{
       fn dot_f32x8_test(u: Vec<f32>, v: Vec<f32>) -> bool {
            let len = cmp::min(u.len(), v.len());
            let u = &u[..len];
            let v = &v[..len];
            approx_eq(dot_f32x8(u, v), dot(u, v))
        }
    }
}
