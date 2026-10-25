"""
The interfaces layer: the command-line entry points.

One module per entry point: it parses and checks its arguments, sets the
process-wide settings, builds the adapters of the infrastructure, calls one
use case and prints what it returns.

The rule. An interfaces module may import every layer of diff_gpt, and no
other layer imports the interfaces. Only an interfaces module has a
`__main__` block. tests/diff_gpt/test_layers.py checks the rule on the source.
"""
