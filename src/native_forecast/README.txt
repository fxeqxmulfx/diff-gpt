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
          augment/       the two offline augmentations: scalers, mixtures,
                         the edits of one series, convolutions, the pipelines
      table.rs           series of one length, as the corpus stores them
      stats.rs           18 features of a series; two-sample Kolmogorov-Smirnov
      model/             DiffGPT's transformer on one sequence
          mod.rs         the sizes and the layout of the parameters; Product,
                         the port for the matrix product, and its definition
          math.rs        exp, RMS normalization, RoPE, causal softmax, SwiGLU,
                         the exclusive projection, each with its backward map
          sequence.rs    the forward pass, the loss and the backward pass
          optimizer.rs   schedule-free AdamW
  crates/application     the use cases and their ports; no thread, no unsafe
      ports.rs           Executor, SeriesReader, SeriesWriter, Split
      corpus.rs          generate the shards; check a source against a reference
  crates/infrastructure  what stands behind the ports
      npy.rs             float32 tables as .npy files
      corpus_files.rs    the reference corpus and this workspace's shards
      threads.rs         the executor on the threads of the process
      kernels.rs         the micro-kernel of the matrix product, in Kernite
      gemm.rs            the blocked product around it, on one thread
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
  taskset -c 0-26,28-54 src/native_forecast/target/release/forecast \
      corpus generate --split holdout --shards 1 --shard-series 4096

`corpus check` compares, for each source, the series this workspace makes
with as many of datasets/tempopfn-synthetic/npy/<split>/<source>.npy, feature
by feature, and fails if a Kolmogorov-Smirnov distance exceeds the critical
one at the level asked for (1e-3 by default). `corpus generate` writes
datasets/tempopfn-rust/<split>/<source>/shard_<k>.npy and skips the shards
that are there. The generators are made first: the two augmented sources read
them back (`--root`, datasets/tempopfn-rust by default, also for the check).
The whole corpus, 16 sources of 65,536 train and 4,096 holdout series of
2,048 steps, takes 45 minutes on 54 threads and 8.6 GB.

The corpus is deterministic
---------------------------

Chunk c (64 series, as TempoPFN's driver asks its generators for them) of a
source of a split is drawn from the stream of the key
[VERSION, split, source, c] (application/src/corpus.rs), so a series depends
neither on the size of the shards, nor on the other sources, nor on the
threads; a series of an augmented source depends besides on the generators'
series it reads, the first N of each in the same split, N being the number
of series the corpus gives a source. The streams are not NumPy's: a port is
checked in distribution, not draw by draw.

The model
---------

The model is `GPT` of src/diff_gpt/domain/model/gpt.py, the transformer of
DiffGPT: tied embedding and readout, pre-norm blocks with parameter-free RMS
normalization, attention with queries and keys normalized per head, RoPE, a
fixed temperature and the exclusive projection (the attended value loses its
component along the token's own value), SwiGLU, no bias anywhere. The domain
states it once, forward and backward, for one sequence (model/sequence.rs);
its tests compare the gradient of every matrix with central differences. All
its matrix products go through the port `Product`: attention makes its
products in bands of 48 rows, since row t needs only its first t + 1 columns.

The infrastructure fills the port with a product blocked as BLIS blocks it
around a 6 x 16 micro-kernel written in Kernite (x86-64-v3: AVX2 and FMA).
On one core of this machine (Xeon E5-2680 v4) it runs at 72 to 86 GFLOPS on
the shapes of a 5M-parameter model at 512 positions, and at 52 on the
narrow products of one attention head:

  cd src/native_forecast
  taskset -c 5 cargo test --release -p diff-gpt-forecast-infrastructure \
      speed -- --ignored --nocapture

Not there yet: the threads that train sequences side by side, the tokens of a
series, the pretraining use case and the scoring.

Sources ported, and how they compare with TempoPFN's
----------------------------------------------------

TempoPFN has 16 sources: 14 generators and 2 offline augmentations of their
output. All are ported, the generators with the parameters that TempoPFN's
continuous_generation.py gives them:

  forecast_pfn, gp, kernel, sawtooth, sinewave, step, anomaly, spike,
  cauker_univariate, ou_process, audio_financial_volatility,
  audio_multi_scale_fractal, audio_network_topology, audio_stochastic_rhythm,
  augmented_per_sample_2048, augmented_temp_batch_2048

Check of 2026-11-11, `corpus check --series 65536` (all 65,536 train series
of the reference against 65,536 of this workspace, critical distance 0.0108
at level 1e-3): every one of the 18 features of every one of the sixteen
sources agrees (the largest distance of gp is 0.0084, of
augmented_per_sample_2048 0.0059, of augmented_temp_batch_2048 0.0074), the
augmented ones made from this workspace's generators. cauker_univariate
makes six series from each random model, so only the first series of each
model is compared: 11,264 against 11,264, critical distance 0.0260. The 4,096
holdout series agree as well (critical distance 0.0431).

gp is the slow source: about 0.7 CPU-seconds a series, 22 minutes for 65,536
series on 54 threads, since each series factors up to a dozen 2048 x 2048
matrices.

What the ports keep of their sources, beyond the documented behaviour:

  augmented_per_sample_2048, augmented_temp_batch_2048
      TempoPFN's offline_per_sample_iid_augmentations.py and
      offline_temp_batch_augmentations.py, with the transforms of its
      src/data/augmentations.py and the scalers of src/data/scalers.py. A
      series comes from a generator chosen by the proportions of the two
      scripts (cauker 2, step and two audio sources 0.1, network topology
      0.5, the rest 1). The first source augments one series at a time: an
      optional scaler, a mixture with up to ten other series before the edits
      and one after, two to six kinds of edits, a random convolution, and it
      starts again until the result differs from the series by 0.05 of its
      interquartile range. The second augments 64 series at once and keeps
      one of those that changed as much. Differences from the scripts: a
      generator's series is taken at a random row of what the corpus holds,
      where the scripts read their files in a cycle; and the frequency and
      the start date of a series, which the calendar edits need and the
      shards do not store, are drawn again from the key of its chunk
      (`Source::calendars`), for forecast_pfn exactly those it was made
      with.
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
