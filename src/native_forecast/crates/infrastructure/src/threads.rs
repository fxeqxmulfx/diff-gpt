//! The executor that runs the jobs of a use case on the threads of this
//! process.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use diff_gpt_forecast_application::ports::Executor;

/// A fixed number of threads, started for each call and joined before it
/// returns. A run restricted with `taskset` gets as many threads as it has
/// CPUs.
#[derive(Clone, Copy, Debug)]
pub struct Threads {
    count: usize,
}

impl Threads {
    /// An executor of `count` threads, at least one.
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self {
            count: count.max(1),
        }
    }

    /// An executor with a thread for each CPU this process may run on.
    #[must_use]
    pub fn available() -> Self {
        Self::new(std::thread::available_parallelism().map_or(1, usize::from))
    }

    /// The number of threads.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }
}

impl Executor for Threads {
    fn map<T, F>(&self, count: usize, job: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Sync,
    {
        let next = AtomicUsize::new(0);
        let results: Mutex<Vec<Option<T>>> = Mutex::new((0..count).map(|_| None).collect());
        std::thread::scope(|scope| {
            for _ in 0..self.count.min(count) {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= count {
                            break;
                        }
                        let result = job(index);
                        results.lock().expect("no job panicked")[index] = Some(result);
                    }
                });
            }
        });
        let results = results.into_inner().expect("no job panicked");
        results
            .into_iter()
            .map(|result| result.expect("every job ran"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_keep_the_order_of_the_jobs() {
        let threads = Threads::new(8);
        let squares = threads.map(1000, |index| index * index);
        assert!(
            squares
                .iter()
                .enumerate()
                .all(|(index, &square)| square == index * index)
        );
        assert!(threads.map(0, |index| index).is_empty());
        assert_eq!(Threads::new(0).count(), 1);
        assert!(Threads::available().count() >= 1);
    }

    #[test]
    fn jobs_run_at_once() {
        // Two jobs that each wait for the other to start can only finish
        // together.
        let barrier = std::sync::Barrier::new(2);
        let done = Threads::new(2).map(2, |index| {
            barrier.wait();
            index
        });
        assert_eq!(done, [0, 1]);
    }
}
