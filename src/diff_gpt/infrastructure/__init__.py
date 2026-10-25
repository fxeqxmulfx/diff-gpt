"""
The infrastructure layer: the code that talks to the OS, databases and processes.

It implements the application's ports and holds what reaches outside the
process: sqlite_data_loader streams series from SQLite for `train` (a
train.DataLoader), numa pins the process to the CPUs of a NUMA node (it reads
/sys and the CPU affinity), train_ddp runs `train` in one DDP worker process
per NUMA node, file_lock, provenance and grokking implement ports.FileLock,
ports.GitHead and ports.BrowserReplay (the browser replay through Node). It
also holds the two adapters that the use cases reach only through files:
leaderboards, the GIFT-Eval and TIME evaluation, with copies of the code it
uses from gift-eval, timebench, gluonts, statsforecast and utilsforecast, and
corpus, the synthetic pretraining corpus, made with TempoPFN's generators
from its clone.

The rule. An infrastructure module imports the domain, the application and
the infrastructure of diff_gpt, never the interfaces, and has no `__main__`
block. tests/diff_gpt/test_layers.py checks the rule on the source.
"""
