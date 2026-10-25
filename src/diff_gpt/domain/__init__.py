"""
The domain layer: pure computation on arrays and tensors.

It holds the tokenizers (encoder_decoder), the GPT model (model), its
optimizer (optimizer), the samplers (sampler), the forecast metrics (metrics)
and the per-window normalization (normalization).

The rule. A domain module imports only the domain of diff_gpt (and numpy,
pandas, torch and the standard library). It starts no process and opens no
database, socket, OS lock or native library: it imports none of subprocess,
sqlite3, socket, fcntl, signal, ctypes, multiprocessing, torch.multiprocessing
and concurrent.futures. It opens no file: it imports none of pathlib, io,
shutil, tempfile, glob, pickle, json, csv, gzip, bz2, lzma, zipfile and
tarfile, uses os only as os.environ, and calls no file reader or writer (open,
torch.save, torch.load, numpy's and pandas' readers and writers). It reads no
environment variable and no CPU affinity, with one exception:
encoder_decoder.DEFAULT_TOKENIZER, the tokenizer of a caller that names none,
from DIFF_GPT_TOKENIZER. It has no `__main__` block.
tests/diff_gpt/test_layers.py checks the rule on the source.
"""
