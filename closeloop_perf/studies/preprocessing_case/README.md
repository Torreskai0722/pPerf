# Fixed-input CenterPoint + ViT preprocessing contention

This single-workload diagnostic compares CPU configuration policies on the RTX
4070 SUPER/i5-14400F, with MPS disabled. Throughput is not an endpoint. The two
pinned model entries and checkpoints come from `../input2_crossed/study.yaml`.
ViT keeps its 512×512 resize. No latency-dependent source selection is allowed.

## Reproduction

Build `Docker/Dockerfile.cuda126` (the older `closeloop_perf/docker` path is
stale). The image installs bpftrace and threadpoolctl. The recorder needs BPF,
performance-monitoring privileges, kernel BTF, and readable scheduler tracefs.
Ubuntu 22.04 bpftrace 0.14 additionally expects `/sys/kernel/debug/tracing`;
inside the privileged container, mount debugfs at `/sys/kernel/debug` if absent.
No tracefs event-enable files are modified by this collector.

Use an isolated checkout with the four packages built inside `pPerf-host`:

```sh
source /opt/ros/humble/setup.bash
cd closeloop_perf
colcon build --symlink-install
source install/setup.bash
```

First record the CPU-only `closeloop_profiler/test/scheduler_sanity_workload.py`
using the same BPF script around Nsight (`nvtx,osrt`, process-tree CPU sampling
and context switches, dwarf backtraces, wait all), export its SQLite trace, and
run `python3 -m closeloop_analyzer.scheduler_evidence SANITY_DIRECTORY`.
Pass the PID namespace inode and eligible CPU bitmask as the BPF script's two
arguments; the regular runner computes these itself. A successful sanity
report requires 50 matched scheduler switches per process, P95 clock residual ≤50 µs,
and ≥99% reconstructed known state. Failed sanity attempts are retained.

Supply explicit paths to the cached input3 `candidates.json`, the pinned model
inventory, and an unused artifact root under `analysis_outputs/preprocessing_case`:

```sh
python3 -m closeloop_experiments.preprocessing_case prepare \
  --inventory INVENTORY --candidates CANDIDATES --artifact-root ARTIFACT_ROOT
python3 -m closeloop_experiments.preprocessing_case run \
  --artifact-root ARTIFACT_ROOT --limit 1
python3 -m closeloop_analyzer.preprocessing_case \
  --artifact-root ARTIFACT_ROOT --baseline-only
python3 -m closeloop_experiments.preprocessing_case run \
  --artifact-root ARTIFACT_ROOT --limit 11
python3 -m closeloop_analyzer.preprocessing_case --artifact-root ARTIFACT_ROOT
```

Preparation verifies original CDRs, source bags and pinned model SHA-256s. It
copies the exact source records for `scene-0770:lidar:198` and
`scene-0770:image:120`, including preprocessing seeds, into provenance. The
existing fixed-input helper writes 40 seconds at LiDAR 20 Hz/camera 12 Hz, with
unique occurrence IDs. Tensor equality is checked twice outside timed regions.
Five warmup inferences precede readiness. ViT starts at 0 s; CenterPoint at 1 s;
replay waits for both. Replay/relay always use CPUs 12–15, four library threads,
queue depth one/best effort, and the existing readiness/completion/drain policy.
GPU clock locks of 3105/10501 MHz are requested, checked after warmup, and reset
on exit; differing readbacks reject causal confirmation.

The authored order is immutable:

1. Default → CPU assignment → Thread limits → Both.
2. Thread limits → Both → Default → CPU assignment.
3. Both → Thread limits → CPU assignment → Default.

Default omits model placement and library limits, inheriting CPUs 0–15. CPU
assignment uses ViT 0–5 and CenterPoint 6–11 (three complete physical P-cores
apiece). Thread limits use three compute threads and PyTorch inter-op one.
Both applies both policies. Inherited thread-tuning environment variables are
removed. Post-warmup snapshots report configured limits separately from OpenCV,
PyTorch, loaded native pool capacities and per-thread affinity; scheduling
activity is measured from events, never inferred from capacity. Pinning can
change library defaults, so these are policy comparisons, not factorial effects.

## Evidence and interpretation

Runs, resolved configurations, source hashes, completion counts, drain
boundaries, clock reset evidence and raw profiles remain under the explicit
artifact root. Failed executions consume their slots and are never overwritten
or retried. The baseline diagnosis must be saved before an intervention starts.
Use one CPU-only sanity workload freely for recorder validation; do not add
measured model runs beyond the three slots per condition.

BPF tracks the launched command by namespace identity, then follows descendants.
Mapping events carry host/container PID–TID pairs. All eligible-CPU switches,
including external occupants, are retained; wakeups, migrations and lifecycle
events cover the experiment process tree. Nsight IDs are normalized before
clock alignment and state reconstruction. Raw event files stay immutable.

The shared analyzer filter computes inclusive P1–P99 cutoffs once from original
completed, non-warmup inference latencies per execution/model/actual scene.
Those retained invocation IDs govern preprocessing, transforms, frame latency,
scheduler attribution and correlations. Repeated payloads remain distinct
occurrences. Full exports, cutoffs, counts, unique sources, excluded occurrences
and retained IDs accompany results. No second crop is applied to pooled plots.

CPU preprocessing means the existing preprocess interval; internal
`data_preprocessor` and GPU work are separate. Frame latency means callback
entry through CUDA-confirmed completion, including decoding. Delivery delay is
separate. Blocked CUDA synchronization is never counted as runnable waiting.
Pool times are summed thread time, not wall duration. Spearman associations
are descriptive within each execution; frame minus preprocessing checks the
part–whole relationship without claiming independence or causation from a
correlation alone.

Tail reporting requires ≥100 retained observations/model/run (also on each
matched subset). Smaller samples retain their identities and median; tails
are withheld, with no replacement run. Plot summaries use median and IQR. Reject causal
attribution on mapping failure, trace loss, <50 matched switches/model, P95
alignment residual >50 µs, or known state coverage <99%. Default→Both is the
primary confirmation and must reduce the diagnosed model's preprocessing
P99−P50 and scheduling-delay endpoint in all three blocks, supported by
identifiable competing threads. CPU-only and thread-only comparisons explain
policy contributions. The two models need not respond equally.

The Matplotlib figure pools the three separately retained distributions and
selects block-one ViT's baseline occurrence nearest its retained preprocessing
P95 among mutually retained Default/Both occurrences, with occurrence-ID tie
breaking. PDF is vector; PNG is 180 dpi. Keep individual results even if pooled
distributions look cleaner. Record file sizes, collection wall time, snapshot
cost and analysis cost. All runs carry identical instrumentation; without an
uninstrumented control, relative timing perturbation is not estimated.

Keep raw bags, profiles, caches, runtime machine paths and AGENTS.md out of Git.
Commit necessary code independently of compact report/figure/provenance.
An inconclusive study must not publish the affirmative manuscript conclusion.
