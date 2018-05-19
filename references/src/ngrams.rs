use std::cmp;

/// Extract n-grams for lengths [n_min, n_max].
///
/// Memory use of the returned Vec:
///
/// * Stack: pointer, capacity, length: 3 machine words
/// * Heap: Each element contains a pointer and a length, so the memory use
///         is m * 2 machine words, where m is the number of n-grams.
pub fn ngrams<T>(mut seq: &[T], n_min: usize, n_max: usize) -> Vec<&[T]> {
    assert!(n_min > 0, "The minimum length of an n-gram is 1");
    assert!(
        n_max >= n_min,
        "The maximum n-gram length should be the minimum length or longer"
    );

    let cap_approx = (n_max - n_min + 1) * seq.len();
    let mut ngrams = Vec::with_capacity(cap_approx);

    while seq.len() >= n_min {
        let upper = cmp::min(n_max, seq.len());

        let mut ngram = &seq[..upper];
        while ngram.len() >= n_min {
            ngrams.push(ngram);
            ngram = &ngram[..ngram.len() - 1];
        }

        seq = &seq[1..];
    }

    ngrams
}

pub fn char_ngrams(seq: &[char], n_min: usize, n_max: usize) -> Vec<&[char]> {
	ngrams(seq, n_min, n_max)
}

#[cfg(test)]
mod tests {
    use super::ngrams;

    #[test]
    fn ngrams_test() {
        let hello_chars: Vec<_> = "hello world".chars().collect();
        let mut hello_check: Vec<&[char]> = vec![
            &['h'],
            &['h', 'e'],
            &['h', 'e', 'l'],
            &['e'],
            &['e', 'l'],
            &['e', 'l', 'l'],
            &['l'],
            &['l', 'l'],
            &['l', 'l', 'o'],
            &['l'],
            &['l', 'o'],
            &['l', 'o', ' '],
            &['o'],
            &['o', ' '],
            &['o', ' ', 'w'],
            &[' '],
            &[' ', 'w'],
            &[' ', 'w', 'o'],
            &['w'],
            &['w', 'o'],
            &['w', 'o', 'r'],
            &['o'],
            &['o', 'r'],
            &['o', 'r', 'l'],
            &['r'],
            &['r', 'l'],
            &['r', 'l', 'd'],
            &['l'],
            &['l', 'd'],
            &['d'],
        ];
        hello_check.sort();

        let mut hello_ngrams = ngrams(&hello_chars, 1, 3);
        hello_ngrams.sort();

        assert_eq!(hello_check, hello_ngrams);
    }

    #[test]
    fn ngrams_23_test() {
        let hello_chars: Vec<_> = "hello world".chars().collect();
        let mut hello_check: Vec<&[char]> = vec![
            &['h', 'e'],
            &['h', 'e', 'l'],
            &['e', 'l'],
            &['e', 'l', 'l'],
            &['l', 'l'],
            &['l', 'l', 'o'],
            &['l', 'o'],
            &['l', 'o', ' '],
            &['o', ' '],
            &['o', ' ', 'w'],
            &[' ', 'w'],
            &[' ', 'w', 'o'],
            &['w', 'o'],
            &['w', 'o', 'r'],
            &['o', 'r'],
            &['o', 'r', 'l'],
            &['r', 'l'],
            &['r', 'l', 'd'],
            &['l', 'd'],
        ];
        hello_check.sort();

        let mut hello_ngrams = ngrams(&hello_chars, 2, 3);
        hello_ngrams.sort();

        assert_eq!(hello_check, hello_ngrams);
    }

    #[test]
    fn empty_ngram_test() {
        let check: &[&[char]] = &[];
        assert_eq!(ngrams::<char>(&[], 1, 3), check);
    }
}
