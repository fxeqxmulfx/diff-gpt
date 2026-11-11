//! Pretraining a model on the synthetic corpus: Track B of
//! `leaderboard_plan.md`, as `src/diff_gpt/application/synthetic_pretrain.py`
//! runs it.
//!
//! The model trains on windows of the train split and is looked at on the
//! holdout split, other series of the same sources: every so many steps
//! the loss of fixed holdout windows is taken at the optimizer's averaged
//! point. Nothing else enters: no benchmark's data is read here.
//!
//! A run keeps in its store:
//! - `run.txt`: its plan and the domain of its tokens, `name = value` lines;
//! - `history.csv`: a row for each evaluation;
//! - `state.npy`: what it goes on from, written at each evaluation: the
//!   two points of the optimizer, its averaged squares, and a row of
//!   counters;
//! - `last.npy`: the averaged parameters, the model to forecast with.
//!
//! A run whose store holds a `run.txt` goes on from its state, if the plan
//! is the one it was started with; only the number of steps may grow.

use std::collections::BTreeMap;

use diff_gpt_forecast_domain::codec::Codec;
use diff_gpt_forecast_domain::model::optimizer::Settings;
use diff_gpt_forecast_domain::model::{Config, Product};
use diff_gpt_forecast_domain::random::Rng;
use diff_gpt_forecast_domain::synth::Source;
use diff_gpt_forecast_domain::table::Table;

use crate::Result;
use crate::ports::{Clock, Crew, Executor, RunStore, SeriesReader, Split};
use crate::training::{State, Trainer};
use crate::windows::{Corpus, Sampler, Windows};

/// What a run is to do.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// The sizes of the model.
    pub model: Config,
    /// The positions of a sequence.
    pub block: usize,
    /// The sequences of a step.
    pub batch: usize,
    /// The steps to take in all.
    pub steps: u64,
    /// The steps between two evaluations.
    pub every: u64,
    /// The holdout windows whose loss an evaluation takes.
    pub holdout: usize,
    /// The settings of the optimizer.
    pub optimizer: Settings,
    /// The norm a gradient is clipped to, if any.
    pub clip: Option<f64>,
    /// The least values of the context that gives a window its units.
    pub least_context: usize,
    /// The windows the domain of the tokens is read from.
    pub domain_windows: usize,
    /// The share of their steps the domain holds.
    pub domain_quantile: f64,
    /// The seed of the initial weights and of the windows.
    pub seed: u64,
}

/// The streams a run draws from, by what they serve.
#[derive(Clone, Copy)]
enum Stream {
    Weights = 1,
    Train = 2,
    Holdout = 3,
    Domain = 4,
}

impl Plan {
    /// The seed of one of the run's streams.
    fn seed_of(&self, stream: Stream) -> u64 {
        Rng::from_key(&[self.seed, stream as u64]).next_u64()
    }

    /// The floating-point operations of a step: twice the multiply-adds of
    /// its matrix products, `GPT.estimate_flops()` times the tokens.
    #[must_use]
    pub fn flops(&self) -> f64 {
        2.0 * self.model.products(self.block) as f64 * (self.block * self.batch) as f64
    }

    /// The plan as `name = value` lines, with the domain of the tokens.
    #[must_use]
    pub fn record(&self, domain: f64) -> String {
        let clip = self
            .clip
            .map_or_else(|| "none".to_owned(), |clip| format!("{clip:?}"));
        let pairs: [(&str, String); 22] = [
            ("vocab", self.model.vocab.to_string()),
            ("width", self.model.width.to_string()),
            ("heads", self.model.heads.to_string()),
            ("layers", self.model.layers.to_string()),
            ("bins", self.model.bins.to_string()),
            ("parameters", self.model.parameters().to_string()),
            ("block", self.block.to_string()),
            ("batch", self.batch.to_string()),
            ("steps", self.steps.to_string()),
            ("every", self.every.to_string()),
            ("holdout", self.holdout.to_string()),
            ("rate", format!("{:?}", self.optimizer.rate)),
            ("warmup", self.optimizer.warmup.to_string()),
            ("beta1", format!("{:?}", self.optimizer.beta1)),
            ("beta2", format!("{:?}", self.optimizer.beta2)),
            ("decay", format!("{:?}", self.optimizer.decay)),
            ("epsilon", format!("{:?}", self.optimizer.epsilon)),
            ("clip", clip),
            ("least_context", self.least_context.to_string()),
            ("domain_windows", self.domain_windows.to_string()),
            ("domain_quantile", format!("{:?}", self.domain_quantile)),
            ("seed", self.seed.to_string()),
        ];
        let mut text = String::new();
        for (name, value) in pairs {
            text += &format!("{name} = {value}\n");
        }
        text + &format!("domain = {domain:?}\n")
    }

    /// The plan and the domain that [`Plan::record`] wrote.
    ///
    /// # Errors
    ///
    /// If a line is missing or does not parse.
    pub fn read(text: &str) -> Result<(Self, f64)> {
        let mut values = BTreeMap::new();
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let (name, value) = line
                .split_once('=')
                .ok_or_else(|| format!("{line:?} is not `name = value`"))?;
            values.insert(name.trim().to_owned(), value.trim().to_owned());
        }
        let text_of = |name: &str| -> Result<&String> {
            values
                .get(name)
                .ok_or_else(|| format!("no `{name}` in the record of the run").into())
        };
        fn parsed<T: core::str::FromStr>(name: &str, value: &str) -> Result<T>
        where
            T::Err: core::fmt::Display,
        {
            value
                .parse()
                .map_err(|error| format!("{name} = {value}: {error}").into())
        }
        macro_rules! take {
            ($name:literal) => {
                parsed($name, text_of($name)?)?
            };
        }
        let clip = text_of("clip")?;
        let plan = Self {
            model: Config {
                vocab: take!("vocab"),
                width: take!("width"),
                heads: take!("heads"),
                layers: take!("layers"),
                bins: take!("bins"),
            },
            block: take!("block"),
            batch: take!("batch"),
            steps: take!("steps"),
            every: take!("every"),
            holdout: take!("holdout"),
            optimizer: Settings {
                rate: take!("rate"),
                warmup: take!("warmup"),
                beta1: take!("beta1"),
                beta2: take!("beta2"),
                decay: take!("decay"),
                epsilon: take!("epsilon"),
            },
            clip: if clip == "none" {
                None
            } else {
                Some(parsed("clip", clip)?)
            },
            least_context: take!("least_context"),
            domain_windows: take!("domain_windows"),
            domain_quantile: take!("domain_quantile"),
            seed: take!("seed"),
        };
        Ok((plan, take!("domain")))
    }

    /// Checks the plan.
    ///
    /// # Errors
    ///
    /// If a size is not valid.
    pub fn check(&self) -> Result<()> {
        self.model.check()?;
        if self.block == 0 || self.batch == 0 || self.every == 0 || self.holdout == 0 {
            return Err(
                "the block, the batch, the interval and the holdout windows are positive".into(),
            );
        }
        if !(self.domain_quantile > 0.0 && self.domain_quantile <= 1.0) || self.domain_windows == 0
        {
            return Err("the domain is a quantile in (0, 1] of some windows".into());
        }
        Ok(())
    }
}

/// The names a run keeps its files under.
pub const RECORD: &str = "run.txt";
/// The history of the evaluations.
pub const HISTORY: &str = "history.csv";
/// What the run goes on from.
pub const STATE: &str = "state.npy";
/// The averaged parameters of the last evaluation.
pub const LAST: &str = "last.npy";

/// The columns of [`HISTORY`].
pub const COLUMNS: &str = "step,flops,train_loss,holdout_loss,norm,seconds";

/// An evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    /// The steps taken.
    pub step: u64,
    /// The floating-point operations of training so far.
    pub flops: f64,
    /// The mean loss of the steps since the last evaluation, each at the
    /// point its gradient was taken.
    pub train_loss: f64,
    /// The loss of the holdout windows at the averaged point.
    pub holdout_loss: f64,
    /// The norm of the last gradient.
    pub norm: f64,
    /// The seconds the run has taken, evaluations included.
    pub seconds: f64,
}

impl Row {
    /// The row as a line of [`HISTORY`].
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "{},{:.4e},{:.5},{:.5},{:.4e},{:.1}\n",
            self.step, self.flops, self.train_loss, self.holdout_loss, self.norm, self.seconds
        )
    }
}

/// The pieces of 16 bits of a number, least first, each a float that holds
/// it exactly.
fn pieces(bits: u64) -> [f32; 4] {
    core::array::from_fn(|piece| ((bits >> (16 * piece)) & 0xffff) as f32)
}

/// The number of those pieces.
fn whole(pieces: &[f32]) -> u64 {
    let pieces = pieces.iter().enumerate();
    pieces.fold(0, |bits, (piece, value)| {
        bits | ((*value as u64) << (16 * piece))
    })
}

/// The state and the seconds of a run as a table: the two points, the
/// averaged squares, and a row that starts with the steps, the bits of the
/// sum of the squared step sizes and the milliseconds, each in four pieces.
/// One table, so that a write keeps all of it or none.
fn pack(state: &State, seconds: f64) -> Table {
    let count = state.y.len();
    let mut values = Vec::with_capacity(4 * count);
    values.extend_from_slice(&state.y);
    values.extend_from_slice(&state.z);
    values.extend_from_slice(&state.squares);
    let counters = [
        state.steps,
        state.total.to_bits(),
        (seconds * 1000.0) as u64,
    ];
    values.extend(counters.iter().flat_map(|counter| pieces(*counter)));
    values.resize(4 * count, 0.0);
    Table::from_values(count, values)
}

/// The state and the seconds that [`pack`] put in a table.
fn unpack(table: &Table, parameters: usize) -> Result<(State, f64)> {
    if table.length() != parameters || table.rows() != 4 || parameters < 12 {
        return Err(format!(
            "the saved state is {} rows of {}, and the model has {parameters} parameters",
            table.rows(),
            table.length()
        )
        .into());
    }
    let counters = table.row(3);
    let state = State {
        y: table.row(0).to_vec(),
        z: table.row(1).to_vec(),
        squares: table.row(2).to_vec(),
        steps: whole(&counters[..4]),
        total: f64::from_bits(whole(&counters[4..8])),
    };
    Ok((state, whole(&counters[8..12]) as f64 / 1000.0))
}

/// The ports of a run.
pub struct Ports<'a, C, E, R, S, K> {
    /// The workers that train.
    pub crew: &'a C,
    /// The executor that reads the corpus.
    pub executor: &'a E,
    /// The corpus.
    pub corpus: &'a R,
    /// Where the run is kept.
    pub store: &'a S,
    /// The clock of this process.
    pub clock: &'a K,
}

/// The codec of a run kept in `store`, and its plan.
///
/// # Errors
///
/// If the store holds no run, or its record does not parse.
pub fn codec_of<S: RunStore>(store: &S) -> Result<(Plan, Codec)> {
    let text = store
        .text(RECORD)?
        .ok_or("there is no run here: no run.txt")?;
    let (plan, domain) = Plan::read(&text)?;
    let codec = Codec::new(plan.model.vocab, domain, plan.model.bins);
    Ok((plan, codec))
}

/// Trains as `plan` says on `sources`, from the state in the store if it
/// holds one, and calls `report` with each evaluation. Each worker gets its
/// matrix product from `make`.
///
/// # Errors
///
/// If the plan is not valid or not the one the stored run was started
/// with, the corpus or the store fails, or the loss stops being finite.
pub fn run<C, E, R, S, K, P>(
    ports: &Ports<'_, C, E, R, S, K>,
    plan: &Plan,
    sources: &[Source],
    make: impl Fn() -> P + Send + Sync + 'static,
    mut report: impl FnMut(&Row),
) -> Result<()>
where
    C: Crew,
    E: Executor,
    R: SeriesReader,
    S: RunStore,
    K: Clock,
    P: Product + Send + Sync,
{
    plan.check()?;
    let store = ports.store;
    let train = Corpus::load(ports.executor, ports.corpus, Split::Train, sources)?;
    let holdout = Corpus::load(ports.executor, ports.corpus, Split::Holdout, sources)?;
    let (domain, state, before) = match store.text(RECORD)? {
        Some(text) => {
            let (started, domain) = Plan::read(&text)?;
            let same = Plan {
                steps: plan.steps,
                ..started.clone()
            };
            if same != *plan {
                return Err(format!(
                    "the run was started with another plan:\n{}and now:\n{}",
                    started.record(domain),
                    plan.record(domain)
                )
                .into());
            }
            let table = store
                .table(STATE)?
                .ok_or("the run has no state yet: start it anew")?;
            let (state, seconds) = unpack(&table, plan.model.parameters())?;
            store.write_text(RECORD, &plan.record(domain))?;
            (domain, state, seconds)
        }
        None => {
            let sampler = Sampler::new(&train, plan.block + 2, plan.least_context)?;
            let seed = plan.seed_of(Stream::Domain);
            let domain = sampler.domain(plan.domain_windows, plan.domain_quantile, seed);
            store.write_text(RECORD, &plan.record(domain))?;
            store.write_text(HISTORY, &format!("{COLUMNS}\n"))?;
            let mut rng = Rng::from_key(&[plan.seed_of(Stream::Weights)]);
            (domain, State::fresh(plan.model.initial(&mut rng)), 0.0)
        }
    };
    let codec = Codec::new(plan.model.vocab, domain, plan.model.bins);
    let least = plan.least_context;
    let windows = Windows::new(
        &train,
        codec,
        plan.block,
        least,
        plan.seed_of(Stream::Train),
    )?;
    let held = Windows::new(
        &holdout,
        codec,
        plan.block,
        least,
        plan.seed_of(Stream::Holdout),
    )?;
    let mut trainer = Trainer::new(
        plan.model,
        plan.block,
        state,
        plan.optimizer,
        plan.clip,
        make,
    );
    let (mut sum, mut count) = (0.0, 0_u64);
    while trainer.state().steps < plan.steps {
        let progress = trainer.step(ports.crew, &windows, plan.batch);
        let step = trainer.state().steps;
        if !progress.loss.is_finite() {
            return Err(format!("the loss of step {step} is {}", progress.loss).into());
        }
        sum += progress.loss;
        count += 1;
        if step.is_multiple_of(plan.every) || step == plan.steps {
            let averaged = trainer.averaged();
            let holdout_loss = trainer.loss(ports.crew, &held, &averaged, (0, plan.holdout));
            let seconds = before + ports.clock.seconds();
            store.write_table(STATE, &pack(trainer.state(), seconds))?;
            store.write_table(LAST, &Table::from_values(averaged.len(), averaged))?;
            let row = Row {
                step,
                flops: plan.flops() * step as f64,
                train_loss: sum / count as f64,
                holdout_loss,
                norm: progress.norm,
                seconds,
            };
            store.append_text(HISTORY, &row.line())?;
            report(&row);
            (sum, count) = (0.0, 0);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use diff_gpt_forecast_domain::model::Plain;

    use super::*;
    use crate::ports::{InTurn, Serial};

    /// A store in memory.
    #[derive(Default)]
    struct Kept {
        texts: RefCell<BTreeMap<String, String>>,
        tables: RefCell<BTreeMap<String, Table>>,
    }

    impl RunStore for Kept {
        fn text(&self, name: &str) -> Result<Option<String>> {
            Ok(self.texts.borrow().get(name).cloned())
        }

        fn write_text(&self, name: &str, text: &str) -> Result<()> {
            self.texts
                .borrow_mut()
                .insert(name.to_owned(), text.to_owned());
            Ok(())
        }

        fn append_text(&self, name: &str, text: &str) -> Result<()> {
            *self.texts.borrow_mut().entry(name.to_owned()).or_default() += text;
            Ok(())
        }

        fn table(&self, name: &str) -> Result<Option<Table>> {
            Ok(self.tables.borrow().get(name).cloned())
        }

        fn write_table(&self, name: &str, table: &Table) -> Result<()> {
            self.tables
                .borrow_mut()
                .insert(name.to_owned(), table.clone());
            Ok(())
        }
    }

    /// A clock that says a second has passed each time it is asked.
    struct Ticks(RefCell<f64>);

    impl Clock for Ticks {
        fn seconds(&self) -> f64 {
            *self.0.borrow_mut() += 1.0;
            *self.0.borrow()
        }
    }

    /// A corpus of sines of several periods, in both splits.
    struct Sines;

    impl SeriesReader for Sines {
        fn series(&self, _: Split, _: Source) -> Result<usize> {
            Ok(24)
        }

        fn read(&self, split: Split, source: Source, first: usize, count: usize) -> Result<Table> {
            let mut table = Table::new(96);
            for series in first..first + count {
                let period = 6.0 + (series % 4) as f64 * 2.0;
                let phase = (series as u64 + split.id() * 100 + source.id()) as f64;
                let wave =
                    |step: usize| (core::f64::consts::TAU * (step as f64 + phase) / period).sin();
                table.push(&(0..96).map(wave).collect::<Vec<_>>());
            }
            Ok(table)
        }
    }

    fn plan(steps: u64) -> Plan {
        Plan {
            model: Config {
                vocab: 32,
                width: 16,
                heads: 2,
                layers: 1,
                bins: 8,
            },
            block: 24,
            batch: 6,
            steps,
            every: 20,
            holdout: 12,
            optimizer: Settings {
                rate: 0.02,
                warmup: 10,
                ..Settings::default()
            },
            clip: Some(5.0),
            least_context: 8,
            domain_windows: 64,
            domain_quantile: 0.999,
            seed: 3,
        }
    }

    fn train(store: &Kept, plan: &Plan) -> Result<Vec<Row>> {
        let ports = Ports {
            crew: &InTurn(2),
            executor: &Serial,
            corpus: &Sines,
            store,
            clock: &Ticks(RefCell::new(0.0)),
        };
        let mut rows = Vec::new();
        run(
            &ports,
            plan,
            &[Source::Sinewave, Source::Sawtooth],
            || Plain,
            |row| rows.push(*row),
        )?;
        Ok(rows)
    }

    #[test]
    fn a_plan_comes_back_from_its_record() {
        let whole = plan(40);
        let text = whole.record(3.25);
        assert_eq!(Plan::read(&text).expect("parsed"), (whole.clone(), 3.25));
        let free = Plan {
            clip: None,
            ..whole
        };
        assert_eq!(Plan::read(&free.record(0.1)).expect("parsed").0, free);
        assert!(Plan::read("vocab = 3\n").is_err());
        assert!(Plan::read(&text.replace("width = 16", "width = wide")).is_err());
        assert!(
            Plan {
                block: 0,
                ..free.clone()
            }
            .check()
            .is_err()
        );
        assert!(
            Plan {
                domain_quantile: 0.0,
                ..free
            }
            .check()
            .is_err()
        );
        // The operations of a step: six a parameter of a matrix and twelve
        // a layer, a width and a position, for each token.
        let model = plan(40).model;
        let per_token = 6 * (model.parameters() - 8 * 16) + 12 * 16 * 24;
        assert_eq!(plan(40).flops(), (per_token * 24 * 6) as f64);
    }

    #[test]
    fn the_counters_of_a_state_survive_the_table() {
        let state = State {
            y: (0..40).map(|index| index as f32).collect(),
            z: vec![1.5; 40],
            squares: vec![0.25; 40],
            steps: 123_456_789_012,
            total: 0.012_345_678_901_234_5,
        };
        let (back, seconds) = unpack(&pack(&state, 98_765.432), 40).expect("the same model");
        assert_eq!(back, state);
        assert!((seconds - 98_765.432).abs() < 1e-3);
        assert!(unpack(&pack(&state, 0.0), 41).is_err());
    }

    #[test]
    fn a_run_learns_and_goes_on_from_its_state() {
        let store = Kept::default();
        let rows = train(&store, &plan(60)).expect("trained");
        assert_eq!(
            rows.iter().map(|row| row.step).collect::<Vec<_>>(),
            [20, 40, 60]
        );
        // Sines are learnt: both losses fall well below a guess, ln 32 = 3.5.
        assert!(
            rows[2].holdout_loss < 0.75 * rows[0].holdout_loss,
            "{rows:?}"
        );
        assert!(
            rows[2].train_loss < 2.0 && rows[2].holdout_loss < 2.0,
            "{rows:?}"
        );
        assert_eq!(rows[1].flops, 2.0 * rows[0].flops);
        assert_eq!(rows[2].seconds, 3.0);
        let history = store.text(HISTORY).expect("read").expect("written");
        assert_eq!(history.lines().count(), 4);
        assert_eq!(history.lines().next(), Some(COLUMNS));
        let (kept, codec) = codec_of(&store).expect("a run");
        assert_eq!((kept, codec.vocab, codec.bins), (plan(60), 32, 8));
        assert!(
            codec.domain() > 0.5 && codec.domain() < 20.0,
            "{}",
            codec.domain()
        );

        // The same run in two parts ends at the same parameters.
        let parts = Kept::default();
        train(&parts, &plan(40)).expect("trained");
        let more = train(&parts, &plan(60)).expect("went on");
        assert_eq!(more.len(), 1);
        assert_eq!(
            (more[0].step, more[0].holdout_loss),
            (60, rows[2].holdout_loss)
        );
        // Two seconds of the first part and one of the second.
        assert_eq!(more[0].seconds, 3.0);
        assert_eq!(
            parts.table(LAST).expect("read"),
            store.table(LAST).expect("read")
        );
        assert_eq!(parts.text(HISTORY).expect("read"), Some(history));

        // Another plan is refused, and a finished run takes no step.
        let other = Plan {
            batch: 7,
            ..plan(80)
        };
        assert!(
            train(&parts, &other)
                .expect_err("refused")
                .to_string()
                .contains("another plan")
        );
        assert!(train(&parts, &plan(60)).expect("nothing to do").is_empty());
    }
}
