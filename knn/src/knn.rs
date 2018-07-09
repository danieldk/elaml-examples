use std::collections::{BTreeMap, BinaryHeap};

use ndarray::prelude::*;
use ordered_float::OrderedFloat;

use {EuclideanDistance, Instance};

/// `KNNBuilder` collects data points for KNN classification.
pub struct KNNBuilder {
    labels: Vec<usize>,
    features: Vec<f32>,
    n_instances: usize,
}

impl Default for KNNBuilder {
    fn default() -> Self {
        KNNBuilder {
            features: Vec::new(),
            labels: Vec::new(),
            n_instances: 0,
        }
    }
}

impl KNNBuilder {
    /// Push a new data point into the builder.
    pub fn push(&mut self, inst: Instance) {
        if self.n_instances != 0 {
            let features_len = self.features.len() / self.n_instances;
            assert_eq!(
                features_len,
                inst.features.len(),
                "Expected a feature vector of size {}, got {}",
                features_len,
                inst.features.len()
            );
        }

        self.n_instances += 1;
        self.features.extend(inst.features);
        self.labels.push(inst.label);
    }
}

/// A K Nearest Neighbor classifier.
pub struct KNN {
    labels: Vec<usize>,
    features: Array2<f32>,
}

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Neighbor {
    distance: OrderedFloat<f32>,
    label: usize,
}

impl From<KNNBuilder> for KNN {
    fn from(builder: KNNBuilder) -> Self {
        let features_len = builder.features.len() / builder.n_instances;

        KNN {
            labels: builder.labels,
            features: Array2::from_shape_vec((builder.n_instances, features_len), builder.features)
                .expect("Number of elements does not correspond to the shape"),
        }
    }
}

impl KNN {
    /// Classify a data point.
    ///
    /// The feature vector and the number of nearest neighbors to used are
    /// specified as arguments. The predicted class is returned.
    pub fn classify(&self, features: &[f32], k: usize) -> usize {
        assert!(k > 0, "k should at least be 1");

        // Compute all distances.
        let distances = self.features
            .euclidean_distance(&ArrayView1::from_shape([features.len()], features).unwrap());

        // Get the nearest neighbors.
        let mut nearest_neighbors = BinaryHeap::with_capacity(k);
        for (idx, &dist) in distances.iter().enumerate() {
            let neighbor = Neighbor {
                label: self.labels[idx],
                distance: OrderedFloat(dist),
            };
            if nearest_neighbors.len() < k {
                nearest_neighbors.push(neighbor);
            } else {
                let mut root = nearest_neighbors
                    .peek_mut()
                    .expect("k > 0, so there should be a neighbor");
                if neighbor.distance < root.distance {
                    *root = neighbor;
                }
            }
        }

        // Count the labels among the nearest neighbors. A BTreeMap
        // is used to ensure stable results.
        let mut label_counts = BTreeMap::new();
        for neighbor in nearest_neighbors {
            let count = label_counts.entry(neighbor.label).or_insert(0);
            *count += 1;
        }

        // Get the most frequent label.
        *label_counts.iter().max_by_key(|kv| kv.1).unwrap().0
    }
}
