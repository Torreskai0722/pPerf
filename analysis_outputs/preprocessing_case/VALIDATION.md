# Validation

Collection and analysis code: `cbf96845268c80b890e54313762ca5ba028af995`.
Figure layout only: `339ac2a` (single column; same samples and timeline selection).

## Committed workspace

Inside `pPerf-host`, a clean `git archive` of the committed Docker recipe, workspace source, studies and README was extracted to a separate directory. It contains no untracked dependencies. All four packages built successfully. The 72 recorded collection-source hashes match committed bytes exactly; all twelve executions used the same source hashes. Each model's preprocessing tensor hash is identical across all twelve runs.

```sh
source /opt/ros/humble/setup.bash
cd closeloop_perf
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select closeloop_profiler closeloop_analyzer closeloop_testbed --event-handlers console_direct+
colcon test --packages-select closeloop_experiments --pytest-args test/test_preprocessing_case.py test/test_runner.py test/test_config.py test/test_package_boundaries.py --event-handlers console_direct+
colcon test-result --verbose
```

Result: **216 tests, 0 errors, 0 failures, 1 skipped**. Tests cover effective-default readback without changing settings, namespace joins/reuse rejection, scheduler reconstruction, clock rejection, loss diagnostics, four policies/order, CUDA-completed frame joins, single-pass scene filtering, and withholding tails below 100 retained observations. The final figure-only changes were exercised by regenerating PDF/PNG from the same retained exports and visually inspecting them. An equality check confirms identical selected frames, P95 and full calling-thread/worker timeline intervals.

The initial unrestricted workspace test invocation reported three unrelated failures because two referenced fixtures are absent from the tracked repository: `studies/mps_two_model_contention_cause/diagnostic_configs/mps2cause-cta-passive-r24.yaml` and `studies/section4_1/study.yaml`. Those failures remain in the local validation logs; this report does not claim that the unrestricted suite passed. An analyzer-only invocation without sourcing the installed workspace failed collection; the correctly sourced run passed. Failed CPU sanity attempts and the successful pre-campaign sanity trace are preserved.

## Recording checks

The CPU-only spawn-based sanity workload passed before model measurements: 74 mapped threads, approximately 2,000 matched switch edges per process, alignment P95 below 2 µs, 100% known-state coverage, no reported trace loss. Nsight process-tree CPU sampling/context-switch capture and the namespace-aware BPF collector were both enabled.

All twelve recorded runs have successful CUDA-completed frame joins and verified source/configuration/raw-file hashes. The final offline entrypoint analyzes all twelve, preserves failures as evidence, and reports **0 runs valid for causal confirmation**. Every ViT execution retains 470 of 480 completions. CenterPoint retains 99–157 observations; the 99-observation execution and matched subsets below 100 have no reported tail statistics. The baseline diagnosis receipt proves it was saved before interventions. Raw runs and authored configurations are read-only.

## Manuscript

The complete eleven-page manuscript was built with `latexmk -pdf -interaction=nonstopmode -halt-on-error`, including BibTeX resolution. The modified subsection has 420 words including its caption (`texcount -sum`). Its text and figure occupy approximately 0.72 of a two-column page in aggregate; the figure floats onto the following page. The vector figure and rendered pages were inspected: labels/legend are readable, there is no clipping, and there are no unresolved references/citations or overfull boxes. Existing unrelated warnings remain: missing author metadata and unrelated float-specifier warnings.

The revised subsection leads with observed policy comparisons and ends with Insight 2. Its protocol and recording diagnostics remain in the experiment report. Existing Elmougy et al. and PyTorch citations are retained. The original Overleaf checkout's unrelated input-data edit is preserved. The publication revision contains the preprocessing subsection, single-column figure, and sequential renumbering of the following two insights.
