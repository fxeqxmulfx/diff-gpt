"""
The dependency rule of the four layers of src/diff_gpt, read off the source.

A module's layer is its folder under src/diff_gpt: domain, application,
infrastructure or interfaces, and the docstring of the layer's __init__.py
states its rule. One test parses every module with `ast`, without importing
it, and names every break of the rule as `path:line: what`.

A second test keeps Track B apart from the legacy benchmarks: the modules
that its entry points import, directly or not, hold none of them, so no
path of a Track B run reaches the data they train on.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "diff_gpt"
LAYERS = ("domain", "application", "infrastructure", "interfaces")
# The layers that each layer may import.
ALLOWED = {
    "domain": {"domain"},
    "application": {"domain", "application"},
    "infrastructure": {"domain", "application", "infrastructure"},
    "interfaces": set(LAYERS),
}
# The modules that domain and application must not import: processes,
# databases, sockets, OS locks and signals, native libraries.
SYSTEM = (
    "subprocess",
    "sqlite3",
    "socket",
    "fcntl",
    "signal",
    "ctypes",
    "multiprocessing",
    "torch.multiprocessing",
    "concurrent.futures",
)
# The names of os that domain and application must not use: the environment
# and the CPU affinity.
ENVIRONMENT = {"environ", "getenv", "putenv", "sched_getaffinity", "sched_setaffinity"}
# The one environment read the domain keeps: the assignment of this name in
# this module.
EXCEPTION = ("domain/encoder_decoder.py", "DEFAULT_TOKENIZER")
# The modules that the domain must not import, since they open files.
FILES = (
    "pathlib",
    "io",
    "shutil",
    "tempfile",
    "glob",
    "pickle",
    "json",
    "csv",
    "gzip",
    "bz2",
    "lzma",
    "zipfile",
    "tarfile",
)
# What the domain must not call, on any object, nor import from torch, numpy
# or pandas: the builtin open and their file readers and writers (and every
# name that starts with read_).
FILE_CALLS = {
    "open",
    "save",
    "load",
    "dump",
    "savez",
    "savez_compressed",
    "savetxt",
    "loadtxt",
    "genfromtxt",
    "fromfile",
    "fromregex",
    "tofile",
    "memmap",
    "open_memmap",
    "from_file",
    "HDFStore",
    "ExcelFile",
    "ExcelWriter",
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_html",
    "to_json",
    "to_latex",
    "to_markdown",
    "to_orc",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "to_stata",
    "to_string",
    "to_xml",
}

type Break = tuple[int, str]  # the line and what breaks the rule there


def _file_call(name: str) -> bool:
    """Whether `name` is one of the file readers and writers."""
    return name in FILE_CALLS or name.startswith("read_")


def _within(module: str, names: tuple[str, ...]) -> bool:
    """Whether `module` is one of `names` or inside one of them."""
    return any(module == name or module.startswith(f"{name}.") for name in names)


def _loaded(node: ast.Import | ast.ImportFrom) -> list[str]:
    """Return the modules that an import may load: `from m import n` may load m.n."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = node.module or ""
    return [module, *(f"{module}.{alias.name}" for alias in node.names)]


def _named_layers(node: ast.Import | ast.ImportFrom) -> list[tuple[str, str | None]]:
    """Return each diff_gpt module that an import names, and its layer or None."""
    if isinstance(node, ast.ImportFrom) and node.module == "diff_gpt":
        return [
            (f"diff_gpt.{alias.name}", alias.name if alias.name in LAYERS else None)
            for alias in node.names
        ]
    if isinstance(node, ast.ImportFrom):
        modules = [node.module or ""]
    else:
        modules = [alias.name for alias in node.names]
    named = []
    for module in modules:
        parts = module.split(".")
        if parts[0] == "diff_gpt":
            layer = parts[1] if len(parts) > 1 and parts[1] in LAYERS else None
            named.append((module, layer))
    return named


def _docstring_only(tree: ast.Module) -> bool:
    """Whether the module holds a docstring and nothing else."""
    return (
        len(tree.body) == 1
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    )


def _is_main_guard(node: ast.AST) -> bool:
    """Whether `node` is `if __name__ == "__main__":`, either way round."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    sides = [node.test.left, *node.test.comparators]
    return any(
        isinstance(side, ast.Name) and side.id == "__name__" for side in sides
    ) and any(
        isinstance(side, ast.Constant) and side.value == "__main__" for side in sides
    )


def _exempt_lines(tree: ast.Module, relative: str) -> set[int]:
    """Return the lines of the one environment read the domain keeps (EXCEPTION)."""
    lines: set[int] = set()
    if relative != EXCEPTION[0]:
        return lines
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(
            isinstance(target, ast.Name) and target.id == EXCEPTION[1]
            for target in targets
        ):
            lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return lines


def _imports(tree: ast.Module, layer: str) -> list[Break]:
    """Return the imports that break the rule: of layers, system modules, files."""
    breaks: list[Break] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        line = node.lineno
        if isinstance(node, ast.ImportFrom) and node.level:
            breaks.append((line, "a relative import"))
            continue
        for module, named in _named_layers(node):
            if named is None:
                breaks.append((line, f"imports {module}, which names no layer"))
            elif named not in ALLOWED[layer]:
                breaks.append((line, f"{layer} imports {named}: {module}"))
        loaded = _loaded(node)
        if layer in {"domain", "application"}:
            breaks += [
                (line, f"{layer} imports {m}") for m in loaded if _within(m, SYSTEM)
            ]
        if layer != "domain":
            continue
        breaks += [
            (line, f"domain imports {m}, which opens files")
            for m in loaded
            if _within(m, FILES)
        ]
        breaks += [
            (line, f"domain imports {m}")
            for m in loaded
            if m.startswith("os.") and m.removeprefix("os.") not in ENVIRONMENT
        ]
        if isinstance(node, ast.ImportFrom) and _within(
            node.module or "", ("torch", "numpy", "pandas")
        ):
            breaks += [
                (line, f"domain imports {alias.name} from {node.module}")
                for alias in node.names
                if _file_call(alias.name)
            ]
    return breaks


def _environment(tree: ast.Module, layer: str, exempt: set[int]) -> list[Break]:
    """Return the uses of os that break the rule: environment, affinity, files."""
    aliases = {"os"} | {
        alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "os" and alias.asname
    }
    breaks: list[Break] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            breaks += [
                (node.lineno, f"{layer} imports os.{alias.name}")
                for alias in node.names
                if alias.name in ENVIRONMENT
            ]
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
            continue
        if node.value.id not in aliases:
            continue
        if node.attr in ENVIRONMENT and node.lineno not in exempt:
            breaks.append((node.lineno, f"{layer} reads os.{node.attr}"))
        if layer == "domain" and node.attr != "environ":
            breaks.append((node.lineno, f"domain uses os.{node.attr}"))
    return breaks


def _calls(tree: ast.Module) -> list[Break]:
    """Return the calls of a file reader or writer, on any object, in the domain."""
    breaks: list[Break] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            name = function.id
        elif isinstance(function, ast.Attribute):
            name = function.attr
        else:
            continue
        if _file_call(name):
            breaks.append((node.lineno, f"domain calls {name}"))
    return breaks


def _breaks(path: Path) -> list[str]:
    """Return every break of the rule in a module of src/diff_gpt: `path:line: what`."""
    relative = path.relative_to(PACKAGE)
    where = f"src/diff_gpt/{relative.as_posix()}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=where)
    breaks: list[Break] = []
    if len(relative.parts) == 1:
        if relative.name != "__init__.py":
            breaks.append((1, "a module outside the layers"))
        elif not _docstring_only(tree):
            breaks.append((1, "holds more than a docstring"))
    elif relative.parts[0] not in LAYERS:
        breaks.append((1, f"{relative.parts[0]} is not a layer"))
    else:
        layer = relative.parts[0]
        if relative.parts[1:] == ("__init__.py",) and not _docstring_only(tree):
            breaks.append((1, "holds more than a docstring"))
        breaks += _imports(tree, layer)
        if layer in {"domain", "application"}:
            breaks += _environment(
                tree, layer, _exempt_lines(tree, relative.as_posix())
            )
        if layer == "domain":
            breaks += _calls(tree)
        if layer != "interfaces":
            breaks += [
                (node.lineno, "a __main__ block outside interfaces")
                for node in ast.walk(tree)
                if _is_main_guard(node)
            ]
    return [f"{where}:{line}: {what}" for line, what in sorted(set(breaks))]


def test_every_module_keeps_the_rule_of_its_layer() -> None:
    """Every module of src/diff_gpt imports, reads and opens what its layer allows."""
    breaks = [
        f"src/diff_gpt/{layer}/__init__.py: missing"
        for layer in LAYERS
        if not (PACKAGE / layer / "__init__.py").is_file()
    ]
    for path in sorted(PACKAGE.rglob("*.py")):
        breaks += _breaks(path)
    assert not breaks, "\n".join(breaks)


TRACK_B = (
    "application.synthetic_pretrain",
    "interfaces.synthetic_pretrain",
    "application.synthetic_rescore",
    "interfaces.synthetic_rescore",
    "application.gift_eval.diff_gpt_forecast",
    "interfaces.gift_eval.diff_gpt_forecast",
    "application.gift_eval.seasonal_naive",
    "interfaces.gift_eval.seasonal_naive",
    "application.gift_eval.files",
)
LEGACY = (
    "anomaly_demo",
    "ett_bench",
    "m4_bench",
    "prob_m4_bench",
    "m4_eval",
    "grokking.series_data",
)


def _module_file(module: str) -> Path | None:
    """Return the file of the diff_gpt module `module`, or None for another one."""
    if module != "diff_gpt" and not module.startswith("diff_gpt."):
        return None
    base = PACKAGE.joinpath(*module.split(".")[1:])
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _closure(modules: tuple[str, ...]) -> set[str]:
    """Return the diff_gpt modules that importing `modules` may load."""
    seen: set[str] = set()
    todo = [f"diff_gpt.{module}" for module in modules]
    while todo:
        module = todo.pop()
        parts = module.split(".")
        for name in (".".join(parts[:end]) for end in range(1, len(parts) + 1)):
            path = _module_file(name)
            if path is None or name in seen:
                continue
            seen.add(name)
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    todo += _loaded(node)
    return seen


def test_track_b_imports_none_of_the_legacy_benchmarks() -> None:
    """No module that a Track B entry point may load is a legacy benchmark."""
    assert all(_module_file(f"diff_gpt.{module}") for module in TRACK_B)
    loaded = _closure(TRACK_B)
    reached = sorted(
        module
        for module in loaded
        for layer in ("application", "interfaces")
        for legacy in LEGACY
        if module == f"diff_gpt.{layer}.{legacy}"
    )
    assert not reached, reached
