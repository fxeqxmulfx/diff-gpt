"""
The code leaderboards runs from gift-eval, timebench, gluonts and statsforecast.

The published scores of GIFT-Eval and TIME were made with gift-eval 3c7be6d,
timebench c633e23, gluonts 0.15.1 and statsforecast 2.1.1, the versions copied
here, with utilsforecast 0.2.17, the base of statsforecast's GroupedArray.
Each subpackage is one project, its modules at the project's own paths, each
a copy of one upstream module cut to what leaderboards' calls run: the
functions, classes and methods they reach, each whole with all its branches
but those that need code left out (each named where it is cut). Each keeps
its upstream license header, if it has one; its docstring names its source
and license, a copy of the license in leaderboards/LICENSE-<project>, and
lists every change from upstream with its reason. A package's `__init__`
holds only a docstring: the copies import each name from the module that
defines it.

Besides those changes, the copies are written in leaderboards' style, which
changes no behavior leaderboards reaches:
- formatted by ruff, imports sorted, `from __future__ import annotations`,
  imports needed only for annotations under TYPE_CHECKING;
- docstrings added, or reworded to a one-line summary; annotations added, in
  the builtin generics and `X | Y` unions of Python 3.10; a type parameter
  (`def batcher[T]`) for a module's TypeVar, and Generator without its
  default send and return types;
- asserts written as `if not <condition>: raise AssertionError(<the same
  message>)`, so that python -O keeps them;
- exception messages put in a variable first; `raise ... from err` inside an
  except clause, and a bare `raise` to re-raise the exception it caught;
- no else or elif after a return, an explicit `return None`, and a value
  returned directly rather than assigned and then returned;
- `raise NotImplementedError` without parentheses;
- boolean parameters keyword-only (every call passes them by keyword);
- zip with an explicit `strict=False`, logging with %-style arguments,
  warnings.warn with its default `stacklevel=1` spelled out;
- `list(x)` for a comprehension that only copies, `next(iter(x))` for
  `list(x)[0]`, `for key in d` for `for key in d.keys()`, `f"{x!r}"` for
  `f"{repr(x)}"`, `()` for `tuple()`, `s.split(sep, maxsplit=1)[0]` for
  `s.split(sep)[0]`, the type of a typing.cast in quotes;
- typing's Protocol and Self in place of typing_extensions';
- a trailing underscore on names that shadow builtins (`input_`, `id_`);
- upstream's TODO notes left out, comments that read as code reworded, and
  blanket `# type: ignore` comments dropped;
- os.path and open() written with pathlib, on the same paths;
- a `# noqa` with its reason where upstream's code is kept as it is (a magic
  number, an unused argument of an interface, a loop variable reassigned).
"""
