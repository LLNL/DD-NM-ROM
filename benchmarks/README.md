# Benchmark helper

`benchmark.py` provides a small launcher for serial workloads and for
SLURM/FLUX batch-script generation. Use the repository virtual environment:

```text
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/matmul.json --backend numpy
```

A workload is either a callable accepting a configuration dictionary, or a
module exposing `prepare(config)` and `run(state)`. Keep construction and
loading in `prepare`; `run` should perform exactly one operation to measure.

For scheduler scripts, generation is side-effect free unless `--submit` is
given:

```text
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/matmul.json --backend torch_gpu \
  --launcher slurm --ranks 1,2,4 \
  --account asccasc --queue pbatch --walltime 01:00:00 \
  --gpus-per-task 1 \
  --output-dir benchmark_jobs
```

The scheduler resource options are available for both SLURM and FLUX:

```text
# SLURM
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/dd_fom_steady.json \
  --backend torch_cpu --launcher slurm \
  --ranks 1,2,4,8,16 \
  --account asccasc --queue pbatch --walltime 01:00:00 \
  --output-dir benchmark_jobs/slurm

# FLUX
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/dd_fom_steady.json \
  --backend torch_gpu --launcher flux \
  --ranks 1,2,4,8,16 \
  --account asccasc --queue pbatch --walltime 01:00:00 \
  --output-dir benchmark_jobs/flux
```

These options populate the account, queue, and walltime directives in the
generated batch scripts. For SLURM, `--cpus-per-task` and
`--gpus-per-task` control the allocation and `srun` task. FLUX follows this
convention by default: `-N <nodes> -x -n <tasks> -g 1 -c 1
-vvv --setopt=mpibind=verbose:1`. Add `--submit` to submit the generated
scripts.

Project DD-FOM/DD-ROM workloads can be added as separate modules using the
same protocol. This keeps benchmark timing independent of pytest setup,
assertions, and profiler output. The rank count must divide the workload's
subdomain count, and GPU parallel jobs should allocate one visible GPU per
rank.

The included DD workloads are in `workloads_dd.py`. They provide
`prepare/run` hooks for the three regression-derived specs and return
`converged`, `newton_iterations`, `residual_norm`, and `internal_timing`
metrics. These appear both in each rank record and in the top-level
`solver_metrics` result field.

Use one spec per decomposition size. Change `n_sub_x` and `n_sub_y` in a
copied spec, then provide only rank counts that divide the total number of
subdomains:

```text
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/dd_fom_steady_8x8.json \
  --backend torch_cpu \
  --launcher slurm \
  --ranks 1,2,4,8,16,32,64
```

Full DD-FOM/DD-ROM solves default to one measured repetition because solver
state may be mutated by a solve.

Generic sweeps are supported without creating temporary spec files. Independent
axes use a Cartesian product:

```json
"sweep": {
  "config.size": [256, 512, 1024],
  "threads": [1, 8]
}
```

This creates six cases. Use `sweep_cases` when values must stay paired, such as
square DD decompositions:

```json
"sweep_cases": [
  {
    "name": "4x4",
    "overrides": {
      "config.n_sub_x": 4,
      "config.n_sub_y": 4
    },
    "ranks": [1, 2, 4, 8, 16]
  },
  {
    "name": "8x8",
    "overrides": {
      "config.n_sub_x": 8,
      "config.n_sub_y": 8
    },
    "ranks": [1, 2, 4, 8, 16, 32, 64]
  }
]
```

Each case/rank combination becomes one local execution or scheduler script.
Rank counts are validated separately for every effective configuration, and
case overrides are passed to workers in memory rather than materialized as
additional JSON files.
