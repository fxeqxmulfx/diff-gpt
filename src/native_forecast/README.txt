native_forecast: the forecasting pipeline of diff-gpt in Rust and Kernite
=========================================================================

A Cargo workspace of its own, beside src/native_transformer. It holds, or will
hold, everything between random numbers and a leaderboard score:

  1. the synthetic corpus: Rust ports of TempoPFN's generators
     (https://github.com/automl/TempoPFN, Apache-2.0, commit 5969ec6);
  2. the pretraining of DiffGPT on that corpus, on this machine's two CPU
     sockets and its GPU;
  3. the zero-shot scoring on GIFT-Eval and fev-bench.

No Python runs anywhere in it. The Python sources of TempoPFN and of
src/diff_gpt are read as the specification; the corpus they made
(datasets/tempopfn-synthetic, ignored by git) is the reference the ports are
checked against.

Layout: clean architecture, each crate depending only on those above it
------------------------------------------------------------------------

  crates/domain          pure computation; no file, thread or clock; no unsafe
      random.rs          xoshiro256++ streams keyed by words, and the samplers
      numeric.rs         linspace, rounding as NumPy, a radix-2 FFT
      linalg.rs          dot products; blocked pivoted Cholesky (as LAPACK dpstrf)
      single.rs          float32 Cholesky that fails as LAPACK spotrf does
      eigen.rs           eigenvectors of a symmetric matrix (dsytrd, then QL)
      calendar.rs        civil dates; TempoPFN's frequencies, start dates and
                         date ranges
      synth/             the sources, one module each; Source names them
          gpytorch.rs    gpytorch's kernels and operators in float32, for gp
          pyo.rs         pyo's oscillators, noises and filters, for the audio
      table.rs           series of one length, as the corpus stores them
      stats.rs           18 features of a series; two-sample Kolmogorov-Smirnov
  crates/application     the use cases and their ports; no thread, no unsafe
      ports.rs           Executor, SeriesReader, SeriesWriter, Split
      corpus.rs          generate the shards; check a source against a reference
  crates/infrastructure  what stands behind the ports
      npy.rs             float32 tables as .npy files
      corpus_files.rs    the reference corpus and this workspace's shards
      threads.rs         the executor on the threads of the process
  crates/interfaces      the `forecast` command: it parses and wires

Commands (from the repository root; CPUs 27 and 55 stay free for the desktop)
-----------------------------------------------------------------------------

  cd src/native_forecast
  cargo fmt --check && cargo clippy --all-targets && cargo test
  cargo build --release
  cd ../..
  taskset -c 0-26,28-54 src/native_forecast/target/release/forecast \
      corpus check --series 65536
  taskset -c 0-26,28-54 src/native_forecast/target/release/forecast \
      corpus generate --split train --shards 16 --shard-series 4096

`corpus check` compares, for each source, the series this workspace makes
with as many of datasets/tempopfn-synthetic/npy/<split>/<source>.npy, feature
by feature, and fails if a Kolmogorov-Smirnov distance exceeds the critical
one at the level asked for (1e-3 by default). `corpus generate` writes
datasets/tempopfn-rust/<split>/<source>/shard_<k>.npy and skips the shards
that are there.

The corpus is deterministic
---------------------------

Chunk c (64 series, as TempoPFN's driver asks its generators for them) of a
source of a split is drawn from the stream of the key
[VERSION, split, source, c] (application/src/corpus.rs), so a series depends
neither on the size of the shards, nor on the other sources, nor on the
threads. The streams are not NumPy's: a port is checked in distribution, not
draw by draw.

Sources ported, and how they compare with TempoPFN's
----------------------------------------------------

TempoPFN has 16 sources: 14 generators and 2 offline augmentations of their
output. Ported so far, each with the parameters that TempoPFN's
continuous_generation.py gives it:

  forecast_pfn, gp, kernel, sawtooth, sinewave, step, anomaly, spike,
  cauker_univariate, ou_process, audio_financial_volatility,
  audio_multi_scale_fractal, audio_network_topology, audio_stochastic_rhythm

These are the 14 generators; the two augmentations are not ported yet.

Check of 2026-11-11, `corpus check --series 65536` (all 65,536 train series
of the reference against 65,536 of this workspace, critical distance 0.0108
at level 1e-3): every one of the 18 features of every one of these fourteen
sources agrees (the largest distance of gp is 0.0084). cauker_univariate
makes six series from each random model, so only the first series of each
model is compared: 11,264 against 11,264, critical distance 0.0260. The 4,096
holdout series agree as well (critical distance 0.0431).

gp is the slow source: about 0.7 CPU-seconds a series, 22 minutes for 65,536
series on 54 threads, since each series factors up to a dozen 2048 x 2048
matrices.

What the ports keep of their sources, beyond the documented behaviour:

  gp
      The series of the reference are shaped by the rounding of gpytorch in
      float32, so the port does that arithmetic entry by entry. The squared
      distances come from the expansion a^2 - 2ab + b^2, whose rounding
      makes the matrix of a short period indefinite; the Cholesky
      factorization then fails and is tried again with 1e-3, 1e-2 and 1e-1
      added to the diagonal, which is white noise of deviation 0.03, 0.1 or
      0.3 in the series (about a quarter of the corpus each); if all fail, the
      draw goes through the eigenvectors with the negative eigenvalues cut.
      A linear kernel added to one dense or product operator needs that
      operator's factor without this last resort: its failure is gpytorch's
      NotPSDError, which TempoPFN answers by starting again with ten times
      the jitter. The periods are drawn around 17 means (np.arange(512,
      1024, 100) has six values), the draws for the unused `exact_freqs`
      and noise level are kept in distribution only, and the fallback of
      TempoPFN after three failures (an RBF draw) is there but not reached.
  audio_*
      pyo 1.1.0 is ported object by object from its C sources (LFO, Sine,
      Noise, BrownNoise, Follower, Metro, TrigExpseg, Biquad): TempoPFN
      records two seconds at 44.1 kHz and keeps the first 2048 samples, 46
      milliseconds, of which TableRec fades in the first 441, and divides
      by the largest magnitude. Within those 46 milliseconds a metronome
      of a longer period fires only its first tick, and an oscillator below
      20 Hz is a piece of its ramp.
  kernel, cauker_univariate
      The reference corpus was made with diff-gpt's stand-ins for TempoPFN's
      modules (src/diff_gpt/infrastructure/corpus/tempopfn): the Gaussian
      processes are drawn through a pivoted Cholesky factorization that
      stops at the numerical rank, and the roots of a CauKer model have
      independent noises. The ports follow the stand-ins.
  forecast_pfn
      The limits on a series are 300, not the 500 the driver asks for; a
      rejected series is redrawn with a frequency taken among eleven, of
      which monthly, quarterly and yearly always fail; the trend counts
      whole days, so it is a staircase within a day. See the module.
