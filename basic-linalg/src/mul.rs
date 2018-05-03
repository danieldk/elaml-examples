pub fn mul(u: &[f32], v: &[f32]) -> Vec<f32> {
    assert_eq!(u.len(), v.len());

    let mut product = vec![0f32; u.len()];

    for i in 0..u.len() {
        product[i] = u[i] * v[i];
    }

    product
}

pub fn mul_inplace(u: &mut [f32], v: &[f32]) {
    assert_eq!(u.len(), v.len());

    for i in 0..u.len() {
        u[i] *= v[i];
    }
}

pub fn mul_slow(u: &[f32], v: &[f32]) -> Vec<f32> {
    assert_eq!(u.len(), v.len());

    let mut product = Vec::with_capacity(u.len());

    for i in 0..u.len() {
        product.push(u[i] * v[i]);
    }

    product
}

#[cfg(test)]
mod tests {
    use super::{mul, mul_inplace, mul_slow};

    #[test]
    fn mul_test() {
        let u = [1f32, -2f32, -3f32];
        let v = [2f32, 4f32, -2f32];

        assert_eq!(&mul(&u, &v), &[2f32, -8f32, 6f32]);
    }

    #[test]
    fn mul_slow_test() {
        let u = [1f32, -2f32, -3f32];
        let v = [2f32, 4f32, -2f32];

        assert_eq!(&mul_slow(&u, &v), &[2f32, -8f32, 6f32]);
    }

    #[test]
    fn mul_inplace_test() {
        let mut u = [1f32, -2f32, -3f32];
        let v = [2f32, 4f32, -2f32];

        mul_inplace(&mut u, &v);

        assert_eq!(&u, &[2f32, -8f32, 6f32]);
    }
}
