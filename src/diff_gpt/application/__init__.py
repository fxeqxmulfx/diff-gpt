"""
The application layer: the use cases, which drive the domain.

It holds `DiffGPT` (diff_gpt), the training loop (train), the in-memory data
loaders (data_loader), `predict_samples_normalized` (normalization),
`paths.ROOT`, the repository folder, and the use cases of the command lines:
the benchmarks (m4_bench, prob_m4_bench, m4_eval, ett_bench, diff_bench,
anomaly_demo, mqar), the grokking experiments (grokking), GIFT-Eval's
forecasters (gift_eval) and Track B's pretraining (synthetic_pretrain,
synthetic_rescore). It owns the ports, the interfaces that the outer layers
implement: train.DataLoader, the protocol of every loader, and the callables
of ports.

The rule. An application module imports only the domain and the application of
diff_gpt (and third-party and standard modules). It starts no process and
opens no database, socket, OS lock or native library: it imports none of
subprocess, sqlite3, socket, fcntl, signal, ctypes, multiprocessing,
torch.multiprocessing and concurrent.futures, and calls none of os.system,
os.popen, os.fork*, os.spawn*, os.exec* and os.posix_spawn*. It reads no
environment variable and no CPU affinity: what it needs of them comes in
through a port or as a value from the command line. It leaves the settings of
the whole process to the command lines, but for torch's seed, the global
random states of torch, CUDA, numpy and Python when a run resumes
(grokking.series, synthetic_pretrain), the filters of diff_gpt's warnings,
and matplotlib's backend in grokking.div97_report.plot_runs and
grokking.recurrence_report.plot: it calls none of torch.set_num_threads,
torch.set_num_interop_threads, torch.set_float32_matmul_precision,
signal.signal, logging.basicConfig, sys.addaudithook and sys.meta_path.insert
or append, and assigns nothing in torch.backends, sys.dont_write_bytecode or
sys.meta_path. It may read and write the files under the paths it is given,
print progress and read clocks and devices. It has no `__main__` block.
tests/diff_gpt/test_layers.py checks the rule on the source.
"""
