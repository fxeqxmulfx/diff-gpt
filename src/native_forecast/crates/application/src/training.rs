//! Training a model: steps of schedule-free AdamW on batches of sequences,
//! the sequences of a batch shared out among the workers of a crew.
//!
//! A worker runs whole sequences, forward and backward, through arrays of
//! its own ([`Workspace`]) and adds their gradients into a vector of its
//! own. A step then has two more turns of the crew, each worker on its
//! stretch of the parameters: the gradients of the workers are summed and
//! their norm is taken, then the optimizer moves the parameters. Which
//! worker runs which sequence changes nothing but the order of the sums.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};

use diff_gpt_forecast_domain::model::optimizer::{self, Schedule, Settings};
use diff_gpt_forecast_domain::model::sequence::Workspace;
use diff_gpt_forecast_domain::model::{Config, Product};

use crate::ports::Crew;

/// A sequence to train on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Example {
    /// The tokens the model reads.
    pub tokens: Vec<u32>,
    /// The level bin of each token, for a model with bins; else empty.
    pub bins: Vec<u32>,
    /// The token to predict at each position.
    pub targets: Vec<u32>,
    /// The weight of each position in the loss; 0 leaves it out.
    pub weights: Vec<f32>,
}

/// Where the sequences come from. Example `index` of batch `batch` is a
/// function of the two numbers alone, so that a run does not depend on the
/// worker that asks.
pub trait Examples: Sync {
    /// Writes the example into `example`, whose vectors it may reuse.
    fn fill(&self, batch: u64, index: usize, example: &mut Example);
}

/// What a worker keeps between sequences.
struct Hand<P> {
    product: P,
    workspace: Workspace,
    gradient: Vec<f32>,
    example: Example,
    /// The loss and the weight of the sequences since the last reset.
    loss: f64,
    weight: f64,
}

impl<P: Product> Hand<P> {
    fn new(config: Config, capacity: usize, product: P) -> Self {
        Self {
            product,
            workspace: Workspace::new(config, capacity),
            gradient: vec![0.0; config.parameters()],
            example: Example::default(),
            loss: 0.0,
            weight: 0.0,
        }
    }

    /// The example the hand holds, through the model: its loss, and its
    /// gradient if asked for.
    fn run(&mut self, parameters: &[f32], backward: bool) {
        let example = &self.example;
        self.workspace.forward(
            &mut self.product,
            parameters,
            &example.tokens,
            &example.bins,
        );
        self.loss += self.workspace.loss(&example.targets, &example.weights);
        self.weight += example
            .weights
            .iter()
            .map(|weight| f64::from(*weight))
            .sum::<f64>();
        if backward {
            self.workspace
                .backward(&mut self.product, parameters, &mut self.gradient);
        }
    }
}

/// What a step reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Progress {
    /// The mean loss of the batch, at the point the gradient was taken.
    pub loss: f64,
    /// The norm of the gradient of that mean, before any clipping.
    pub norm: f64,
}

/// What a run goes on from: the two points of the optimizer, its averaged
/// squares, and its count of steps with the sum of their squared sizes.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    /// Where the gradients are taken.
    pub y: Vec<f32>,
    /// The point that takes the plain steps.
    pub z: Vec<f32>,
    /// The averaged squared gradients.
    pub squares: Vec<f32>,
    /// The steps taken.
    pub steps: u64,
    /// The sum of the squared step sizes.
    pub total: f64,
}

impl State {
    /// The state before the first step, at `parameters`.
    #[must_use]
    pub fn fresh(parameters: Vec<f32>) -> Self {
        Self {
            z: parameters.clone(),
            squares: vec![0.0; parameters.len()],
            y: parameters,
            steps: 0,
            total: 0.0,
        }
    }
}

/// A model in training.
pub struct Trainer<P> {
    config: Config,
    capacity: usize,
    make: Box<dyn Fn() -> P + Send + Sync>,
    state: State,
    schedule: Schedule,
    clip: Option<f64>,
    gradient: Vec<f32>,
    hands: Vec<RwLock<Option<Hand<P>>>>,
}

impl<P: Product + Send + Sync> Trainer<P> {
    /// A trainer of the model of sizes `config` on sequences of up to
    /// `capacity` tokens, which goes on from `state`. Each worker gets its
    /// product from `make`; a gradient longer than `clip` is scaled down to
    /// it.
    ///
    /// # Panics
    ///
    /// If the sizes are not valid or the state is not of this model.
    pub fn new(
        config: Config,
        capacity: usize,
        state: State,
        settings: Settings,
        clip: Option<f64>,
        make: impl Fn() -> P + Send + Sync + 'static,
    ) -> Self {
        config.check().expect("valid sizes");
        let count = config.parameters();
        assert_eq!(
            (state.y.len(), state.z.len(), state.squares.len()),
            (count, count, count),
            "the state of another model"
        );
        Self {
            config,
            capacity,
            make: Box::new(make),
            schedule: Schedule::resumed(settings, state.steps, state.total),
            state,
            clip,
            gradient: vec![0.0; count],
            hands: Vec::new(),
        }
    }

    /// The sizes of the model.
    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// The most tokens a sequence may have.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// The state to go on from.
    #[must_use]
    pub const fn state(&self) -> &State {
        &self.state
    }

    /// The parameters to evaluate the model at: the optimizer's average.
    #[must_use]
    pub fn averaged(&self) -> Vec<f32> {
        let mut x = vec![0.0; self.state.y.len()];
        let beta1 = self.schedule.settings().beta1;
        optimizer::averaged(&self.state.y, &self.state.z, beta1, &mut x);
        x
    }

    /// A hand for each worker of the crew.
    fn staff(&mut self, crew: &impl Crew) {
        while self.hands.len() < crew.size() {
            self.hands.push(RwLock::new(None));
        }
    }

    /// Runs `count` examples of batch `batch` through the model at
    /// `parameters`, shared out among the workers, each into its hand.
    fn share<C: Crew, E: Examples>(
        &self,
        crew: &C,
        examples: &E,
        parameters: &[f32],
        (batch, count): (u64, usize),
        backward: bool,
    ) {
        let next = AtomicUsize::new(0);
        crew.all(&|worker| {
            let mut slot = self.hands[worker].write().expect("no worker panicked");
            let hand =
                slot.get_or_insert_with(|| Hand::new(self.config, self.capacity, (self.make)()));
            hand.loss = 0.0;
            hand.weight = 0.0;
            if backward {
                hand.gradient.fill(0.0);
            }
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= count {
                    break;
                }
                examples.fill(batch, index, &mut hand.example);
                hand.run(parameters, backward);
            }
        });
    }

    /// The loss and the weight of what the hands last ran.
    fn tally(&self) -> (f64, f64) {
        let mut sums = (0.0, 0.0);
        for hand in &self.hands {
            if let Some(hand) = hand.read().expect("no worker panicked").as_ref() {
                sums = (sums.0 + hand.loss, sums.1 + hand.weight);
            }
        }
        sums
    }

    /// One step on a batch of `count` examples: the batch of the number of
    /// steps taken so far.
    ///
    /// # Panics
    ///
    /// If the batch is empty or has no weight.
    pub fn step<C: Crew, E: Examples>(&mut self, crew: &C, examples: &E, count: usize) -> Progress {
        assert!(count > 0, "an empty batch");
        self.staff(crew);
        let batch = self.schedule.steps();
        self.share(crew, examples, &self.state.y, (batch, count), true);
        let (loss, weight) = self.tally();
        assert!(weight > 0.0, "a batch without weight");

        // The gradient of the mean: each worker sums its stretch of the
        // hands' gradients.
        let size = crew.size();
        let stretch = self.gradient.len().div_ceil(size);
        let share = (1.0 / weight) as f32;
        let hands = &self.hands;
        let squares: Vec<Mutex<f64>> = (0..size).map(|_| Mutex::new(0.0)).collect();
        let slots: Vec<Mutex<&mut [f32]>> =
            self.gradient.chunks_mut(stretch).map(Mutex::new).collect();
        crew.all(&|worker| {
            let Some(slot) = slots.get(worker) else {
                return;
            };
            let mut sum = slot.lock().expect("no worker panicked");
            sum.fill(0.0);
            let first = worker * stretch;
            for hand in hands {
                let hand = hand.read().expect("no worker panicked");
                let Some(hand) = hand.as_ref().filter(|hand| hand.weight > 0.0) else {
                    continue;
                };
                for (total, part) in sum.iter_mut().zip(&hand.gradient[first..]) {
                    *total += part;
                }
            }
            let mut square = 0.0_f64;
            for total in sum.iter_mut() {
                *total *= share;
                square += f64::from(*total) * f64::from(*total);
            }
            *squares[worker].lock().expect("no worker panicked") = square;
        });
        drop(slots);
        let norm = squares
            .iter()
            .map(|square| *square.lock().expect("no worker panicked"))
            .sum::<f64>()
            .sqrt();

        // The step of the optimizer, each worker on its stretch.
        let clip = self.clip;
        let factors = self.schedule.advance();
        let state = &mut self.state;
        let stretches = state
            .y
            .chunks_mut(stretch)
            .zip(state.z.chunks_mut(stretch))
            .zip(state.squares.chunks_mut(stretch))
            .zip(self.gradient.chunks_mut(stretch));
        let slots: Vec<_> = stretches.map(Mutex::new).collect();
        crew.all(&|worker| {
            let Some(slot) = slots.get(worker) else {
                return;
            };
            let mut slot = slot.lock().expect("no worker panicked");
            let (((y, z), squares), gradient) = &mut *slot;
            if let Some(most) = clip {
                optimizer::clip(gradient, norm, most);
            }
            factors.apply(y, z, squares, gradient);
        });
        drop(slots);
        state.steps = self.schedule.steps();
        state.total = self.schedule.total();
        Progress {
            loss: loss / weight,
            norm,
        }
    }

    /// The mean loss of `count` examples of batch `batch` at `parameters`.
    ///
    /// # Panics
    ///
    /// If the examples have no weight.
    pub fn loss<C: Crew, E: Examples>(
        &mut self,
        crew: &C,
        examples: &E,
        parameters: &[f32],
        (batch, count): (u64, usize),
    ) -> f64 {
        self.staff(crew);
        self.share(crew, examples, parameters, (batch, count), false);
        let (loss, weight) = self.tally();
        assert!(weight > 0.0, "examples without weight");
        loss / weight
    }
}

#[cfg(test)]
mod tests {
    use diff_gpt_forecast_domain::model::Plain;
    use diff_gpt_forecast_domain::random::Rng;

    use super::*;
    use crate::ports::InTurn;

    const CONFIG: Config = Config {
        vocab: 12,
        width: 16,
        heads: 2,
        layers: 2,
        bins: 0,
    };

    /// Sequences that count up from a random start, modulo the vocabulary:
    /// the next token follows from the last.
    struct Counting {
        length: usize,
    }

    impl Examples for Counting {
        fn fill(&self, batch: u64, index: usize, example: &mut Example) {
            let mut rng = Rng::from_key(&[7, batch, index as u64]);
            let start = rng.index(CONFIG.vocab);
            let token = |position: usize| ((start + position) % CONFIG.vocab) as u32;
            example.tokens = (0..self.length).map(token).collect();
            example.targets = (1..=self.length).map(token).collect();
            example.weights = vec![1.0; self.length];
            example.bins.clear();
        }
    }

    fn trainer(clip: Option<f64>) -> Trainer<Plain> {
        let settings = Settings {
            rate: 0.02,
            warmup: 10,
            ..Settings::default()
        };
        let state = State::fresh(CONFIG.initial(&mut Rng::from_key(&[1])));
        Trainer::new(CONFIG, 10, state, settings, clip, || Plain)
    }

    #[test]
    fn training_learns_to_count() {
        let examples = Counting { length: 10 };
        let mut trainer = trainer(None);
        let crew = InTurn(3);
        let start = trainer.averaged();
        let before = trainer.loss(&crew, &examples, &start, (1_000_000, 16));
        // An untrained model knows nothing: the loss is that of a guess.
        assert!(
            (before - (CONFIG.vocab as f64).ln()).abs() < 0.05,
            "{before}"
        );
        let mut last = Progress {
            loss: 0.0,
            norm: 0.0,
        };
        for _ in 0..150 {
            last = trainer.step(&crew, &examples, 8);
        }
        assert_eq!(trainer.state().steps, 150);
        assert!(last.norm > 0.0 && last.loss < 0.5, "{last:?}");
        let end = trainer.averaged();
        let after = trainer.loss(&crew, &examples, &end, (1_000_000, 16));
        assert!(after < 0.3, "the loss went from {before} to {after}");
    }

    #[test]
    fn a_step_does_not_depend_on_the_size_of_the_crew() {
        let examples = Counting { length: 10 };
        let (mut one, mut many) = (trainer(None), trainer(None));
        for _ in 0..3 {
            let alone = one.step(&InTurn(1), &examples, 7);
            let shared = many.step(&InTurn(5), &examples, 7);
            assert!((alone.loss - shared.loss).abs() < 1e-6);
            assert!((alone.norm - shared.norm).abs() < 1e-5 * alone.norm);
        }
        let pairs = one.state().y.iter().zip(&many.state().y);
        let apart = pairs.fold(0.0_f32, |worst, (a, b)| worst.max((a - b).abs()));
        // The first steps of Adam are the size of the rate whatever the
        // gradient: sums in another order may differ in the last places.
        assert!(apart < 2e-4, "the parameters are {apart} apart");
        assert_eq!(one.state().steps, 3);
        assert!((one.state().total - many.state().total).abs() < 1e-18);
    }

    #[test]
    fn a_long_gradient_is_clipped_to_the_norm_asked_for() {
        let examples = Counting { length: 10 };
        // With no clipping and with a bound far above the gradient the
        // steps are the same; with a small bound the squares are smaller.
        let (mut free, mut loose, mut tight) =
            (trainer(None), trainer(Some(1e9)), trainer(Some(1e-3)));
        let crew = InTurn(2);
        let norm = free.step(&crew, &examples, 4).norm;
        assert_eq!(loose.step(&crew, &examples, 4).norm, norm);
        assert_eq!(tight.step(&crew, &examples, 4).norm, norm);
        assert_eq!(free.state().squares, loose.state().squares);
        let sum = |trainer: &Trainer<Plain>| -> f64 {
            let squares = trainer.state().squares.iter();
            squares.map(|square| f64::from(*square)).sum()
        };
        let ratio = sum(&tight) / sum(&free);
        let expected = (1e-3 / norm).powi(2);
        assert!(
            (ratio / expected - 1.0).abs() < 1e-3,
            "{ratio} against {expected}"
        );
    }
}
