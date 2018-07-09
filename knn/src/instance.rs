use std::io::{self, BufRead, Lines};

use failure::{err_msg, Error};

/// A data instance.
#[derive(Debug, PartialEq, PartialOrd)]
pub struct Instance {
    /// The data point label.
    pub label: usize,

    /// The features of the data point.
    pub features: Vec<f32>,
}

/// An iterator over data points.
pub struct InstanceIter<R> {
    lines: Lines<R>,
}

impl<R> InstanceIter<R>
where
    R: BufRead,
{
    /// Construct a new iterator over data points.
    pub fn new(buf_read: R) -> Self {
        InstanceIter {
            lines: buf_read.lines(),
        }
    }
}

impl<R> Iterator for InstanceIter<R>
where
    R: BufRead,
{
    type Item = Result<Instance, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(process_line(self.lines.next()?))
    }
}

fn process_line(line: Result<String, io::Error>) -> Result<Instance, Error> {
    let line = line?;

    let mut iter = line.split_whitespace();

    // Get and parse label.
    let label_str = iter.next().ok_or(err_msg("Line is missing label"))?;
    let label = label_str.parse::<usize>()?;

    // Parse the remaining columns as features.
    let features = iter.map(|v| v.parse::<f32>()).collect::<Result<_, _>>()?;

    Ok(Instance { label, features })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{Instance, InstanceIter};

    #[test]
    fn iter_test() {
        let lines = "1 1.0 -1.0 0.0 2.0\n0 -1.0 1.0 1.0 -1.0";
        let mut iter = InstanceIter::new(Cursor::new(lines));

        assert_eq!(
            iter.next().unwrap().unwrap(),
            Instance {
                label: 1,
                features: vec![1.0, -1.0, 0.0, 2.0],
            }
        );
        assert_eq!(
            iter.next().unwrap().unwrap(),
            Instance {
                label: 0,
                features: vec![-1.0, 1.0, 1.0, -1.0],
            }
        );
        assert!(iter.next().is_none());
    }
}
