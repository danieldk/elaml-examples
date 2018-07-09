#[cfg(test)]
#[macro_use]
extern crate approx;
extern crate clap;
extern crate failure;
extern crate ndarray;
extern crate num_traits;
extern crate ordered_float;
extern crate stdinout;

use std::fs::File;
use std::io::BufReader;

use stdinout::OrExit;

mod args;
use args::parse_args;

pub mod distance;
use distance::EuclideanDistance;

mod evaluation;
use evaluation::Evaluator;

mod instance;
use instance::{Instance, InstanceIter};

mod knn;
use knn::{KNNBuilder, KNN};

fn main() {
    let matches = parse_args();

    let k = matches
        .value_of("knearest")
        .map(|v| v.parse().or_exit("k is not a valid integer", 1))
        .unwrap_or(3);

    let train_file =
        File::open(matches.value_of("TRAIN").unwrap()).or_exit("Cannot open training file", 1);
    let test_file =
        File::open(matches.value_of("TEST").unwrap()).or_exit("Cannot open test file", 1);

    let mut builder = KNNBuilder::default();
    for instance in InstanceIter::new(BufReader::new(train_file)) {
        let instance = instance.or_exit("Cannot read instance", 1);
        builder.push(instance);
    }

    let model: KNN = builder.into();

    let mut eval = Evaluator::default();

    for instance in InstanceIter::new(BufReader::new(test_file)) {
        let instance = instance.or_exit("Cannot read instance", 1);
        let predicted = model.classify(&instance.features, k);

        eval.count(instance.label, predicted);
    }

    println!("Accuracy: {:.1}", eval.accuracy() * 100.);
}
