//! The crew of a training run: a thread for each worker, each bound to a
//! processor of its own.
//!
//! Worker `w` is always on the same processor, so the memory it touches
//! first, the arrays it runs sequences through, lies on that processor's
//! socket and stays in reach. The threads are started for each call and
//! joined before it returns, as the executor of [`crate::threads`] does;
//! a thread binds itself with a launch of Kernite's grid on its one
//! processor, which leaves the caller there.

use std::collections::BTreeSet;
use std::fs;

use diff_gpt_forecast_application::Result;
use diff_gpt_forecast_application::ports::Crew;
use kernite_cpu::Grid;

/// A worker on each of some processors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pinned {
    cpus: Vec<usize>,
}

/// A number the system keeps in a file of a processor's topology.
fn topology(cpu: usize, name: &str) -> Result<usize> {
    let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/{name}");
    let text = fs::read_to_string(&path).map_err(|error| format!("{path}: {error}"))?;
    Ok(text
        .trim()
        .parse()
        .map_err(|error| format!("{path}: {error}"))?)
}

impl Pinned {
    /// A worker on each of `cpus`.
    #[must_use]
    pub const fn on(cpus: Vec<usize>) -> Self {
        Self { cpus }
    }

    /// A worker for each core the process may run on, on the first of the
    /// core's processors: two threads of one core share its arithmetic, and
    /// a second one adds little to a matrix product.
    ///
    /// # Errors
    ///
    /// If the system does not say which processors the process may run on,
    /// or which cores they belong to.
    pub fn cores() -> Result<Self> {
        let allowed = kernite_cpu::cpus().map_err(|error| error.to_string())?;
        let mut seen = BTreeSet::new();
        let mut cpus = Vec::new();
        for cpu in allowed {
            let core = (
                topology(cpu, "physical_package_id")?,
                topology(cpu, "core_id")?,
            );
            if seen.insert(core) {
                cpus.push(cpu);
            }
        }
        Ok(Self { cpus })
    }

    /// The crew with its first `count` workers only, one at least.
    #[must_use]
    pub fn first(mut self, count: usize) -> Self {
        self.cpus.truncate(count.max(1));
        self
    }

    /// The processor of each worker.
    #[must_use]
    pub fn cpus(&self) -> &[usize] {
        &self.cpus
    }
}

impl Crew for Pinned {
    fn size(&self) -> usize {
        self.cpus.len()
    }

    fn all(&self, job: &(dyn Fn(usize) + Sync)) {
        std::thread::scope(|scope| {
            for (worker, cpu) in self.cpus.iter().copied().enumerate() {
                scope.spawn(move || {
                    kernite_cpu::launch(Grid::on(&[cpu]), |_| {}).unwrap_or_else(|error| {
                        panic!("worker {worker} on processor {cpu}: {error}")
                    });
                    job(worker);
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// The processors the calling thread may run on, from the system.
    fn allowed() -> Vec<usize> {
        let status = fs::read_to_string("/proc/thread-self/status").expect("a Linux process");
        let line = status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .expect("the list of processors");
        let mut cpus = Vec::new();
        for range in line.trim().split(',') {
            let (first, last) = range.split_once('-').unwrap_or((range, range));
            cpus.extend(
                first.parse::<usize>().expect("a number")..=last.parse().expect("a number"),
            );
        }
        cpus
    }

    #[test]
    fn each_worker_runs_on_its_processor() {
        let crew = Pinned::cores().expect("the topology");
        assert!(crew.size() >= 1);
        // No two workers share a core: the cores are as many as the workers.
        let cores: BTreeSet<_> = crew
            .cpus()
            .iter()
            .map(|cpu| {
                let package = topology(*cpu, "physical_package_id").expect("a package");
                (package, topology(*cpu, "core_id").expect("a core"))
            })
            .collect();
        assert_eq!(cores.len(), crew.size());
        let seen = Mutex::new(vec![Vec::new(); crew.size()]);
        for _ in 0..2 {
            crew.all(&|worker| {
                seen.lock().expect("no panic")[worker].push(allowed());
            });
        }
        let seen = seen.into_inner().expect("no panic");
        for (worker, cpu) in crew.cpus().iter().enumerate() {
            assert_eq!(seen[worker], [vec![*cpu], vec![*cpu]], "worker {worker}");
        }
        // The caller stays where it was.
        assert!(allowed().len() >= crew.size());
        let few = crew.clone().first(1);
        assert_eq!((few.size(), few.cpus()[0]), (1, crew.cpus()[0]));
        assert_eq!(Pinned::on(vec![3, 5]).size(), 2);
    }
}
