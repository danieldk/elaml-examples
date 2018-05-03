pub fn dot_unrolled(mut u: &[f32], mut v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len());

    let (mut dp0, mut dp1, mut dp2, mut dp3, mut dp4, mut dp5, mut dp6, mut dp7) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    while u.len() >= 8 {
        dp0 += u[0] * v[0];
        dp1 += u[1] * v[1];
        dp2 += u[2] * v[2];
        dp3 += u[3] * v[3];
        dp4 += u[4] * v[4];
        dp5 += u[5] * v[5];
        dp6 += u[6] * v[6];
        dp7 += u[7] * v[7];

        u = &u[8..];
        v = &v[8..];
    }

    dp0 + dp1 + dp2 + dp3 + dp4 + dp5 + dp6 + dp7 + dot(u, v)
}

pub fn dot(u: &[f32], v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len());

    let mut dp = 0f32;

    for i in 0..u.len() {
        dp += u[i] * v[i]
    }

    dp
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use super::{dot, dot_unrolled};

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-1
    }

    #[test]
    fn dot_simple_test() {
        let u = [1f32, -2f32, -3f32];
        let v = [2f32, 4f32, -2f32];
        let w = [-1f32, 3f32, 2.5f32];

        assert_approx_eq!(dot(&u, &v), 0f32);
        assert_approx_eq!(dot(&u, &w), -14.5f32);
        assert_approx_eq!(dot(&v, &w), 5f32);
    }

    quickcheck!{
       fn dot_unrolled_test(u: Vec<f32>, v: Vec<f32>) -> bool {
            let len = cmp::min(u.len(), v.len());
            let u = &u[..len];
            let v = &v[..len];
            approx_eq(dot_unrolled(u, v), dot(u, v))

        }
    }
}
