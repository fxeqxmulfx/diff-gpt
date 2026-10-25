"""
Stand-in for TempoPFN's `src.synthetic_generation.cauker.cauker_generator`.

CauKer draws each series from a random structural causal model: a DAG whose
roots are GP draws, of composite kernels and random means, and whose other
nodes are random activations of affine maps of their parents. TempoPFN draws
the roots with cupy's multivariate_normal on the GPU, an eigendecomposition by
cusolver, after cp.random.seed(seed) with the series' seed, so all roots of a
series take the same white noise: the last bits of their kernel matrices set
their correlation (|rho| 0.39 at the median over pairs of roots, 0.78 at the
90th percentile), and two roots of one kernel and mean are one series. The
original CauKer (github.com/ShifengXIE/CauKer, CauKer.py at ed00670) passes
no seed there, and draws its roots independently.

This is TempoPFN's `CauKerGenerator` (Apache-2.0, a copy of the license in
synthetic/LICENSE-TempoPFN), changed in two places:
- each root draws its noise from a generator of its own, seeded by the
  series' seed and the root: independent roots, as in the original;
- the roots are drawn on the CPU, by `sample_gaussian` from the matrix of
  `lag_kernel_matrix`, the KernelSynth stand-in's, in about 0.15 s each.
Its other draws come from a RandomState and a Random of the series' seed,
where TempoPFN seeds numpy's and Python's global generators with it: the same
streams, so a seed gives TempoPFN's DAG, kernels, means and activations, and
only the noise of the roots differs. It needs no GPU, nor cupy.
"""

from __future__ import annotations

import functools
import random
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
)

from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.kernel_synth.kernel_synth import (
    lag_kernel_matrix,
    sample_gaussian,
)

if TYPE_CHECKING:
    from sklearn.gaussian_process.kernels import Kernel

    from src.synthetic_generation.generator_params import CauKerGeneratorParams


class CauKerGenerator(AbstractTimeSeriesGenerator):
    """
    Structural-Causal-Model GP-based time series generator.

    This class is a refactor of the original script-level implementation, exposing
    the same logic as instance methods and generating one multivariate series per call.
    """

    def __init__(self, params: CauKerGeneratorParams) -> None:
        """Keep `params`; seed the generators of the draws with their global seed."""
        self.params = params
        self.seed_generators(params.global_seed)

    def seed_generators(self, seed: int) -> None:
        """Seed the generators of all draws but the roots' noise with `seed`."""
        self.np_random = np.random.RandomState(seed)
        self.random = random.Random(seed)  # noqa: S311 - a simulation, not secrets.

    def build_kernel_bank(self, time_length: int) -> list[Kernel]:
        """Return the base kernels, their periods in steps of `time_length`."""
        return [
            # Hourly and sub-hourly cycles
            ExpSineSquared(periodicity=24 / time_length),
            ExpSineSquared(periodicity=48 / time_length),
            ExpSineSquared(periodicity=96 / time_length),
            # Hourly components embedded in weekly structure
            ExpSineSquared(periodicity=24 * 7 / time_length),
            ExpSineSquared(periodicity=48 * 7 / time_length),
            ExpSineSquared(periodicity=96 * 7 / time_length),
            # Daily and sub-daily cycles
            ExpSineSquared(periodicity=7 / time_length),
            ExpSineSquared(periodicity=14 / time_length),
            ExpSineSquared(periodicity=30 / time_length),
            ExpSineSquared(periodicity=60 / time_length),
            ExpSineSquared(periodicity=365 / time_length),
            ExpSineSquared(periodicity=365 * 2 / time_length),
            # Weekly, monthly and quarterly variations
            ExpSineSquared(periodicity=4 / time_length),
            ExpSineSquared(periodicity=26 / time_length),
            ExpSineSquared(periodicity=52 / time_length),
            ExpSineSquared(periodicity=4 / time_length),
            ExpSineSquared(periodicity=6 / time_length),
            ExpSineSquared(periodicity=12 / time_length),
            ExpSineSquared(periodicity=4 / time_length),
            ExpSineSquared(periodicity=(4 * 10) / time_length),
            ExpSineSquared(periodicity=10 / time_length),
            # Stationary and noise kernels
            DotProduct(sigma_0=0.0),
            DotProduct(sigma_0=1.0),
            DotProduct(sigma_0=10.0),
            RBF(length_scale=0.1),
            RBF(length_scale=1.0),
            RBF(length_scale=10.0),
            RationalQuadratic(alpha=0.1),
            RationalQuadratic(alpha=1.0),
            RationalQuadratic(alpha=10.0),
            WhiteKernel(noise_level=0.1),
            WhiteKernel(noise_level=1.0),
            ConstantKernel(),
        ]

    def random_binary_map(self, a: Kernel, b: Kernel) -> Kernel:
        """Combine two kernels with + or *, at random."""
        binary_ops = [lambda x, y: x + y, lambda x, y: x * y]
        return self.np_random.choice(binary_ops)(a, b)

    def zero_mean(self, x: np.ndarray) -> np.ndarray:
        """Return the zero mean."""
        return np.zeros_like(x)

    def linear_mean(self, x: np.ndarray) -> np.ndarray:
        """Return a x + b, for a and b uniform on [-1, 1]."""
        a = self.np_random.uniform(-1.0, 1.0)
        b = self.np_random.uniform(-1.0, 1.0)
        return a * x + b

    def exponential_mean(self, x: np.ndarray) -> np.ndarray:
        """Return a exp(b x), for a and b uniform on [0.5, 1.5]."""
        a = self.np_random.uniform(0.5, 1.5)
        b = self.np_random.uniform(0.5, 1.5)
        return a * np.exp(b * x)

    def anomaly_mean(self, x: np.ndarray) -> np.ndarray:
        """Return zeros but for 1 to 5 spikes, uniform on [-5, 5], at random steps."""
        m = np.zeros_like(x)
        num_anomalies = self.np_random.randint(1, 6)
        for _ in range(num_anomalies):
            idx = self.np_random.randint(0, len(x))
            m[idx] += self.np_random.uniform(-5.0, 5.0)
        return m

    def random_mean_combination(self, x: np.ndarray) -> np.ndarray:
        """Return the sum or the product of two random means of the library."""
        mean_functions = [
            self.zero_mean,
            self.linear_mean,
            self.exponential_mean,
            self.anomaly_mean,
        ]
        m1, m2 = self.np_random.choice(mean_functions, 2, replace=True)
        combine_ops = [lambda u, v: u + v, lambda u, v: u * v]
        return self.np_random.choice(combine_ops)(m1(x), m2(x))

    def sample_from_gp_prior(
        self,
        *,
        kernel: Kernel,
        X: np.ndarray,
        mean_vec: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw from N(mean_vec, kernel(X)) on the grid `X`, the noise from `rng`."""
        return mean_vec + sample_gaussian(lag_kernel_matrix(kernel, X), rng)

    def generate_random_dag(self, num_nodes: int, max_parents: int = 3) -> nx.DiGraph:
        """Return a random DAG: each node has up to `max_parents` earlier parents."""
        G = nx.DiGraph()
        nodes = list(range(num_nodes))
        self.random.shuffle(nodes)
        G.add_nodes_from(nodes)
        for i in range(num_nodes):
            possible_parents = nodes[:i]
            num_par = self.np_random.randint(
                0, min(len(possible_parents), max_parents) + 1
            )
            for p in self.random.sample(possible_parents, num_par):
                G.add_edge(p, nodes[i])
        return G

    def random_activation(self, x: np.ndarray, func_type: str = "linear") -> np.ndarray:
        """Apply the activation `func_type` to `x`, of random parameters."""
        if func_type == "linear":
            a = self.np_random.uniform(0.5, 2.0)
            b = self.np_random.uniform(-1.0, 1.0)
            return a * x + b
        if func_type == "relu":
            return np.maximum(0.0, x)
        if func_type == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if func_type == "sin":
            return np.sin(x)
        if func_type == "mod":
            c = self.np_random.uniform(1.0, 5.0)
            return np.mod(x, c)
        # Anything else: leaky ReLU
        alpha = self.np_random.uniform(0.01, 0.3)
        return np.where(x > 0, x, alpha * x)

    def random_edge_mapping(self, parents_data: list[np.ndarray]) -> np.ndarray:
        """Return a random activation of a random affine map of the parents."""
        combined = np.stack(parents_data, axis=1)
        W = self.np_random.randn(len(parents_data))
        b = self.np_random.randn()
        non_linear_input = combined @ W + b
        chosen_func = self.np_random.choice(
            ["linear", "relu", "sigmoid", "sin", "mod", "leakyrelu"]
        )
        return self.random_activation(non_linear_input, chosen_func)

    def generate_scm_time_series(
        self,
        *,
        time_length: int,
        max_parents: int,
        seed: int,
        num_nodes: int,
    ) -> dict[int, np.ndarray]:
        """Return the series of each node of the random SCM of `seed`."""
        self.seed_generators(seed)

        dag = self.generate_random_dag(num_nodes, max_parents=max_parents)
        kernel_bank = self.build_kernel_bank(time_length)

        root_nodes = [n for n in dag.nodes if dag.in_degree(n) == 0]
        node_data: dict[int, np.ndarray] = {}

        X = np.linspace(0.0, 1.0, time_length)

        # Sample roots directly from the GP prior, each root with its own noise
        for r in root_nodes:
            selected_kernels = self.np_random.choice(
                kernel_bank, self.np_random.randint(1, 8), replace=True
            )
            kernel = functools.reduce(self.random_binary_map, selected_kernels)
            mean_vec = self.random_mean_combination(X)
            # A child of the series' SeedSequence, whose stream no
            # default_rng(int) shares: [seed, 0] would be default_rng(seed),
            # which draws a range of num_channels in generate_time_series.
            noise = np.random.SeedSequence(seed, spawn_key=(r,))
            node_data[r] = self.sample_from_gp_prior(
                kernel=kernel, X=X, mean_vec=mean_vec, rng=np.random.default_rng(noise)
            )

        # Propagate through DAG
        for node in nx.topological_sort(dag):
            if node in root_nodes:
                continue
            parents = list(dag.predecessors(node))
            parents_ts = [node_data[p] for p in parents]
            node_data[node] = self.random_edge_mapping(parents_ts)

        return node_data

    def generate_time_series(self, random_seed: int | None = None) -> np.ndarray:
        """Generate one multivariate series with shape (length, num_channels)."""
        seed = self.params.global_seed if random_seed is None else random_seed

        # Resolve num_channels which can be int or (min, max)
        desired_channels: int | tuple[int, int] = self.params.num_channels
        if isinstance(desired_channels, tuple):
            low, high = desired_channels
            if low > high:
                low, high = high, low
            num_channels = int(np.random.default_rng(seed).integers(low, high + 1))
        else:
            num_channels = int(desired_channels)

        if num_channels > self.params.num_nodes:
            msg = (
                f"num_channels ({num_channels}) cannot exceed "
                f"num_nodes ({self.params.num_nodes})."
            )
            raise ValueError(msg)

        node_data = self.generate_scm_time_series(
            time_length=self.params.length,
            max_parents=self.params.max_parents,
            seed=seed,
            num_nodes=self.params.num_nodes,
        )

        chosen_nodes = self.random.sample(list(node_data.keys()), num_channels)
        channels = [node_data[n].astype(np.float32) for n in chosen_nodes]
        return np.stack(channels, axis=1)  # (length, num_channels)
