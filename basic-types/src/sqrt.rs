use std::f32;

const SQRT_N_ITERATIONS: usize = 5;

fn fast_sqrt(v: f32) -> f32 {
    // Initial square root approximation:
    // http://h14s.p5r.org/2012/09/0x5f3759df.html
    let mut i = v.to_bits();
    i = 0x1fbd1df5 + (i >> 1);
    f32::from_bits(i)
}

pub fn sqrt(v: f32) -> f32 {
    if v == 0f32 || v.is_nan() || v.is_infinite() {
        return v;
    }

    if v < 0f32 {
        return f32::NAN;
    }

    let mut x = fast_sqrt(v);

    for _ in 0..SQRT_N_ITERATIONS {
        x = x - (x * x - v) / (2. * x);
    }

    x
}

#[cfg(test)]
mod tests {
    use std::f32;

    use super::sqrt;

    const TOLERANCE: f32 = 1e-4;

    quickcheck! {
        fn sqrt_test(v: f32) -> bool {
            if v < 0f32 {
                sqrt(v).is_nan()
            } else if v.is_infinite() {
                sqrt(v).is_infinite()
            } else {
                // Check against sqrt from the standard library.
                (sqrt(v) - v.sqrt()).abs() < TOLERANCE
            }
        }
    }

    #[test]
    fn sqrt_smallest_test() {
        // Someone (I think Lukas Stein), asked whether a small value
        // wouldn't cause infinity or NaN because of a division by zero.
        // This test is to show that does not happen for normalized
        // numbers. The reason that it works (for reasonable initial
        // guesses) is that x * x can still be represented as a
        // denormalized number. If the guess is much too small, x
        // will typically be much larger due to a relatively large
        // numerator and a relatively small denominator.
        assert!(sqrt(f32::MIN_POSITIVE) < TOLERANCE);
    }
}
