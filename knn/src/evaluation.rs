pub struct Evaluator {
    n_instances: usize,
    n_correct: usize,
}

impl Default for Evaluator {
    fn default() -> Self {
        Evaluator {
            n_instances: 0,
            n_correct: 0,
        }
    }
}

impl Evaluator {
    pub fn count(&mut self, correct: usize, predicted: usize) {
        self.n_instances += 1;

        if predicted == correct {
            self.n_correct += 1;
        }
    }

    pub fn accuracy(&self) -> f32 {
        self.n_correct as f32 / self.n_instances as f32
    }
}
