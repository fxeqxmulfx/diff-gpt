"""
The application layer: the use cases, which drive the domain.

It holds `DiffGPT` (diff_gpt), the training loop (train), the in-memory data
loaders (data_loader), `predict_samples_normalized` (normalization) and
`paths.ROOT`, the repository folder. It owns the ports, the interfaces that
the outer layers implement: train.DataLoader, the protocol of every loader.

The rule. An application module imports only the domain and the application
of diff_gpt (and third-party and standard modules). It starts no process and
opens no database, socket, OS lock or native library: it imports none of
subprocess, sqlite3, socket, fcntl, signal, ctypes, multiprocessing,
torch.multiprocessing and concurrent.futures. It reads no environment
variable and no CPU affinity: what it needs of them comes in through a port or
as a value from the command line. It may read and write the files under the
paths it is given, print progress and read clocks and devices. It has no
`__main__` block. tests/diff_gpt/test_layers.py checks the rule on the source.
"""
