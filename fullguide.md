# Black Hole Lab — Roadmap and Build-Along Textbook

> **This guide focuses on what we will build next, not re-explaining code you already understand.**
>
> Each implementation chapter will contain: **goal → why it matters → mathematics/physics → full code for that step → code walkthrough → run commands → tests/experiments → interpretation → definition of done.**
>
> Future code will be written when we reach each milestone and after checking the repository's actual files. We will not dump an untested future codebase all at once.

Repository: [sahandkhodayi/Black-Hole](https://github.com/sahandkhodayi/Black-Hole)

---

# 1. Destination: What We Are Building

The goal is a scientific-computing and machine-learning project investigating light bending around a nonrotating Schwarzschild black hole.

By the end, the project should:
1. Simulate light-ray trajectories numerically.
2. Validate the solver against mathematical and physical expectations.
3. Generate a reproducible dataset.
4. Train a conventional ML model as a baseline.
5. Implement a Physics-Informed Neural Network (PINN) using the governing differential equation.
6. Compare models fairly on unseen cases.
7. Investigate model errors, especially in difficult strong-bending regimes.
8. Publish reproducible experiments, plots, metrics, and a technical report.

**The PINN is not the whole project.** A PINN trained on unreliable simulations would not produce a meaningful scientific result.

## Main research question

> Does adding the photon trajectory differential equation to a neural network's training objective improve physical consistency or generalization compared with a data-only neural network?

We will not assume the answer is yes. The experiments must be allowed to prove either result.

---

# 2. What You Have Already Built

The current prototype includes a numerical solver for the reduced Schwarzschild photon-orbit equation, closest-approach initial conditions, an escape event, a symmetric trajectory, and plots comparing different closest approaches.

You already understand the solver state (u, du/dphi), iterative numerical integration, closest approach, and the observed trend that closer escaping rays bend more strongly in your tested examples.

**We will not spend the next session re-teaching these existing lines.** We will make the simulator reliable and use it as the foundation for the rest of the project.

Current limitations to address:
- experiments run at module import time,
- calculation and printed output are mixed,
- termination metadata is not represented cleanly,
- deflection angle needs a separate tested function,
- there is no real automated test suite,
- numerical convergence and physical validation are incomplete,
- plot-saving behavior and messages need to agree,
- clean-environment setup needs verification.

---

# 3. Complete Milestone Map

## Milestone 1 — Make the simulator reliable
**Deliverable:** clean reusable API, explicit result/status metadata, separate deflection-angle function, automated tests, numerical convergence study.

## Milestone 2 — Validate the physics and critical trajectories
**Deliverable:** weak-field comparison, careful investigation near the photon sphere, clear handling of escaping and unresolved trajectories, reproducible plots.

## Milestone 3 — Build a reproducible dataset
**Deliverable:** generator that stores input parameters, targets, termination status, units, solver settings, and metadata.

## Milestone 4 — Establish a baseline
**Deliverable:** a simple reference method and a fixed evaluation protocol before deep learning.

## Milestone 5 — Train a conventional neural network
**Deliverable:** PyTorch regression model, train/validation/test splits, saved checkpoint, learning curves, and error analysis.

## Milestone 6 — Implement a PINN
**Deliverable:** neural network approximating the trajectory, automatic first/second derivatives, physics-residual loss, and initial/boundary-condition loss.

## Milestone 7 — Compare methods fairly
**Deliverable:** common evaluation pipeline comparing numerical solver, baseline, normal neural network, and PINN.

## Milestone 8 — Choose one research extension
**Deliverable:** one evidence-driven extension such as inverse parameter estimation, adaptive sampling, or learning near-critical trajectories.

## Milestone 9 — Publish
**Deliverable:** reproducible commands, figures, metrics, limitations, clean setup instructions, and a technical report.

---

# 4. Standard Format for Every Chapter

Each chapter will follow this order:

1. **Goal:** one concrete feature.
2. **Why:** what problem it solves and why the project needs it.
3. **Math/physics:** equations, variables, units, assumptions, and a derivation when useful.
4. **Files:** what will be created or modified.
5. **Full code:** complete contents of each file changed in that chapter, not disconnected snippets.
6. **Code explanation:** important lines, APIs, data flow, and design decisions. No re-teaching programming basics you already know unless they matter to a new concept.
7. **Run it:** exact commands and expected behavior.
8. **Tests/experiments:** automated checks and scientific validation as appropriate.
9. **Interpretation:** what the result means and what it does not prove.
10. **Definition of done:** criteria to satisfy before moving on.

We will inspect the current repository before each code change. The code for a future chapter depends on earlier decisions and results, so writing it all in advance would risk creating an untested guide that does not fit the actual project.

---

# MILESTONE 1 — Make the Simulator Reliable

## Chapter 1.1 — Separate the library from experiments

### Goal
Importing the geodesics module should define reusable code without automatically running simulations or printing output.

### Why
Tests, scripts, and the future dataset generator must be able to import the simulator safely. Experiment loops belong in an executable script, not at module scope.

### Files
- Modify src/lab/geodesics.py.
- Clean up or add src/simulate.py as the experiment entry point.

### Full-code deliverable
We will refactor the actual files into reusable functions in geodesics.py and a main() function in simulate.py, guarded by the standard Python main check. The complete replacement contents of each changed file will be shown when we implement this chapter.

### Explanation
We will cover library module vs executable script, import-time side effects, the main guard, and separation of computation from presentation.

### Verification
Run the simulation script and confirm it still produces the expected experiment. Then import the module and confirm that no experiment is run and no output is printed.

### Done when
The same simulation can be called from a script, a test, or a future dataset generator without side effects.

---

## Chapter 1.2 — Give the simulation an explicit result object

### Goal
Return enough information to understand how integration ended, not only arrays of trajectory samples.

### Why
A trajectory alone may not reveal whether the ray escaped, reached the integration limit, or encountered a numerical failure. The dataset generator must not silently label every run as a valid escaping ray.

### Design
The result should include:
- trajectory samples,
- termination status,
- escape angle when the escape event occurs,
- solver success/message,
- requested closest-approach parameter,
- solver settings needed for reproducibility.

We will choose exact field names after inspecting the code during implementation.

### Math/numerical concept
The escape event is a root of a function evaluated along the numerical solution. Here the event function is g(phi, y) = u. A decreasing zero crossing identifies the escaping branch in the current setup.

Do not confuse:
- **event detected** — the chosen event condition was found,
- **solver success** — the integrator completed according to its criteria,
- **physical classification** — our interpretation of the simulation.

These are related but not identical.

### Full-code deliverable
We will update the result type, adapt the solver, and update the calling script. Each changed file will be shown in full and explained.

### Tests
A standard escaping case records an event; a run that reaches the angular limit is not mislabeled as escaped; solver failure information is retained; trajectory arrays remain aligned.

### Done when
Every result makes its termination behavior explicit.

---

## Chapter 1.3 — Separate deflection-angle calculation

### Goal
Make the deflection angle a separate, documented, tested calculation instead of printing a formula from inside the simulation.

### Math
For the symmetric scattering setup, if phi_escape is the angular distance from closest approach to the asymptotic outgoing direction,

\[
\alpha = 2\phi_{\mathrm{escape}}-\pi.
\]

The result is in radians; use numpy.degrees for degrees. The factor of two accounts for both symmetric halves, and subtracting pi compares the total angular change with a straight-line path.

Closest-approach radius is not the same thing as impact parameter. Weak-field formulas commonly use impact parameter, so later comparisons must convert definitions correctly.

### Full-code deliverable
Add a deflection_angle function, update the result/CLI interface, and print clearly labelled escape angle and deflection angle. The full changed files will be included in the chapter.

### Tests
Check the zero-deflection limit, factor of two, radians/degrees conversion, and behavior when no escape angle exists.

### Done when
Deflection is an explicit tested quantity, not something inferred from console output.

---

## Chapter 1.4 — Add a real pytest suite

### Goal
Make regressions visible when we change the solver.

### Planned files
~~~text
tests/
├── test_geodesics.py
├── test_deflection.py
└── test_simulation.py
~~~

### Full-code deliverable
We will add complete test files based on the actual function signatures after Chapters 1.1–1.3. We will not create empty files just to match a tree.

### Test coverage
- the right-hand side returns derivatives in the correct order,
- inverse-radius conversion behaves correctly,
- invalid inputs raise clear errors,
- trajectory arrays have matching lengths,
- a known escaping case detects the event,
- the deflection formula and units are correct,
- reaching the maximum integration angle without an event is handled honestly.

Run:

~~~bash
python -m pytest -q
~~~

If pytest is missing, install it in the active environment and record it as a development dependency.

### Important distinction
Unit tests verify software behavior; they do not by themselves prove the physics is correct. That requires independent numerical/physical validation.

### Done when
The suite runs and would catch common bugs such as reversed derivatives or a missing factor of two.

---

## Chapter 1.5 — Measure numerical convergence

### Goal
Determine whether the reported deflection is stable when solver settings change.

### Why
A smooth plot does not prove that the measured value is accurate.

### Experiment
Keep the initial condition fixed and compare settings such as:

| Run | rtol | atol |
|---|---:|---:|
| A | 1e-7 | 1e-9 |
| B | 1e-9 | 1e-11 |
| C | 1e-11 | 1e-13 |

Then vary max_step separately. Change one factor at a time.

### Math
For successive estimates alpha_1, alpha_2, alpha_3, inspect

\[
\Delta_{21}=|\alpha_2-\alpha_1|,
\qquad
\Delta_{32}=|\alpha_3-\alpha_2|.
\]

Decreasing differences suggest stabilization, but do not prove correctness by themselves. Check event behavior and solver status as well.

### Full-code deliverable
Create a convergence script that runs the same case with different settings, saves a results table, reports differences, and optionally plots deflection against settings. Full script and explanation will be included.

### Done when
We can justify the solver settings for dataset generation with evidence.

---

# MILESTONE 2 — Validate the Physics and Explore Critical Trajectories

## Chapter 2.1 — Validate weak deflection

### Goal
Check the model when a ray passes far from the black hole.

### Math
In the weak-field limit, leading-order Schwarzschild deflection is

\[
\alpha \approx \frac{2r_s}{b},
\]

where b is the impact parameter and the ray is far from the strong-field critical region.

### Important issue
Our simulation starts at closest approach, not at infinity with a specified impact parameter. We must derive or numerically establish the relationship between these quantities before comparing them. They are not interchangeable.

### Full-code deliverable
Create a validation script that runs large closest-approach cases, computes numerical deflection, compares it with a correctly converted weak-field estimate, and plots the difference.

### Done when
The comparison uses consistent definitions and units, and discrepancies are discussed rather than hidden.

---

## Chapter 2.2 — Investigate the photon sphere

### Goal
Explore behavior near the critical scattering regime.

For a Schwarzschild black hole,

\[
r_{\mathrm{ph}}=\frac{3}{2}r_s.
\]

This is the photon-sphere radius. The current input restriction is not a complete capture/scattering classifier.

### Experiments
- vary the initial condition near the critical regime,
- record angular integration length and event status,
- check whether the ray spends a long interval near the photon sphere,
- identify where numerical behavior becomes unreliable,
- distinguish escape, capture, unresolved, and solver-failure states.

A run that merely reaches max_angle must not be called captured.

### Full-code deliverable
Create a parameter sweep that records status and numerical diagnostics and plots deflection against a clearly defined parameter.

### Done when
The project honestly distinguishes validated escaping rays from incomplete integrations.

---

## Chapter 2.3 — Make plots and experiments reproducible

### Goal
Produce report-quality plots that can be regenerated from recorded settings.

### Full-code deliverable
Build plotting functions that can compare rays, label the event horizon and photon sphere, keep units consistent, exclude infinite-radius samples, save to a requested path, and return figure/axes when useful.

Save experiment settings with outputs. The complete plotting implementation will be provided when this chapter is reached.

### Explanation topics
Separate visualization from computation; polar-to-Cartesian conversion; masks for infinite values; axis limits; reproducible figures.

### Done when
Save messages match real files and plots can be recreated.

---

# MILESTONE 3 — Build a Reproducible Dataset

## Chapter 3.1 — Define the learning target

The first ML target will be **scalar deflection regression**: given a well-defined ray parameter, predict the numerical deflection angle for validated escaping rays.

Later, we can try trajectory prediction, where the model takes parameters and angular coordinate and predicts u(phi). That task is more complex because points from one trajectory are correlated.

Every feature and target needs a clear mathematical meaning, units, and provenance.

## Chapter 3.2 — Implement a dataset generator

### Planned structure
~~~text
src/
└── lab/
    ├── geodesics.py
    ├── dataset.py
    └── ...
scripts/
└── generate_dataset.py
data/
└── ...
~~~

### Full-code deliverable
Create a generator and CLI that accept a parameter range and output path, run the simulator, record successful and unresolved cases, save inputs and targets, and store generation settings/schema metadata. Use CSV for a simple scalar table or a suitable NumPy format for arrays.

### Done when
A fresh run produces understandable data and enough metadata for another person to regenerate it.

## Chapter 3.3 — Inspect data and prevent leakage

Check missing/non-finite values, duplicates, target ranges and units, parameter coverage, and counts by status.

If a trajectory contributes many points, do not scatter points from the same trajectory across training and test sets if the goal is generalization to unseen trajectories. Split by trajectory or parameter case as appropriate.

### Full-code deliverable
Add a data-inspection script and reproducible train/validation/test split. Save split assignments or random seed.

### Done when
The test set measures the intended generalization question.

---

# MILESTONE 4 — Establish a Baseline

Before deep learning, implement a simple reference: a justified regression model, interpolation over a validated numerical table, or weak-field approximation in its valid regime.

For predictions alpha_hat_i and reference values alpha_i,

\[
\mathrm{MAE}=\frac{1}{N}\sum_i|\hat\alpha_i-\alpha_i|,
\]

\[
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_i(\hat\alpha_i-\alpha_i)^2}.
\]

MAE summarizes typical absolute error; RMSE penalizes larger errors more strongly. We will report units and inspect failure cases too.

### Full-code deliverable
Implement the selected baseline, evaluate it on held-out cases, and save prediction/reference and residual plots.

### Done when
There is a reproducible baseline and fixed evaluation protocol.

---

# MILESTONE 5 — Train a Conventional Neural Network

## Chapter 5.1 — First PyTorch regression model

### Math
A network predicts y_hat_i = f_theta(x_i). Mean-squared-error loss is

\[
\mathcal{L}_{\mathrm{data}}=\frac{1}{N}\sum_i(\hat y_i-y_i)^2.
\]

An optimizer updates network parameters theta to reduce the loss.

### Full-code deliverable
Create complete runnable files for data loading, preprocessing, a small PyTorch model, training and validation loops, checkpoint saving, metric calculation, and prediction/learning-curve plots.

### Walkthrough
Explain tensor shapes, the model forward pass, loss, optimizer, gradient updates, train/eval modes, validation without gradients, checkpoint save/load, and how preprocessing avoids leakage.

### Experiments
Keep the test set out of model selection; fit normalization only on training data; record seeds/configuration; compare held-out predictions with the baseline.

### Done when
A clean run trains, saves, and evaluates the model without manual notebook edits.

## Chapter 5.2 — Analyze model errors

Create a report script for MAE/RMSE, predictions versus references, residuals, errors by parameter regime, worst predictions, and comparison with the baseline.

An acceptable average error can hide severe failures near the photon sphere or at the edge of training coverage.

### Done when
We can explain where the model succeeds, fails, and whether it generalizes beyond interpolation.

---

# MILESTONE 6 — Build the PINN

## Chapter 6.1 — Derive the physics residual

The trajectory equation is

\[
\frac{d^2u}{d\phi^2}+u-\frac{3}{2}u^2=0.
\]

Let the network approximate u_theta(phi). Its residual is

\[
\mathcal{R}_\theta(\phi)=
\frac{d^2u_\theta}{d\phi^2}
+u_\theta-\frac{3}{2}u_\theta^2.
\]

The physics loss samples angular positions phi_j:

\[
\mathcal{L}_{\mathrm{physics}}=
\frac{1}{N_f}\sum_j|\mathcal{R}_\theta(\phi_j)|^2.
\]

The equation alone does not uniquely select the desired trajectory. Initial/boundary conditions are necessary; data may also be used.

## Chapter 6.2 — Learn automatic differentiation

### Goal
Calculate first and second derivatives of a network output with respect to its input using PyTorch.

### Full-code deliverable
First create a small standalone exercise: define a network, create an input tensor with gradients enabled, compute the output and its first/second derivatives, then compare with a known analytic function.

### Walkthrough
Explain tensor shapes, torch.autograd.grad, graph construction for second derivatives, and how to verify derivative results.

### Done when
You can explain where the derivatives come from and verify them on a known function.

## Chapter 6.3 — Implement PINN and losses

A combined objective can be

\[
\mathcal{L}=
\lambda_d\mathcal{L}_{\mathrm{data}}
+\lambda_f\mathcal{L}_{\mathrm{physics}}
+\lambda_b\mathcal{L}_{\mathrm{conditions}}.
\]

Data loss matches selected numerical samples; physics loss penalizes equation residual; condition loss enforces initial/boundary conditions. The weights lambda control each term.

For the current parameterization, if phi = 0 is closest approach, conditions include u(0) = u0 and u'(0) = 0.

### Full-code deliverable
Build complete files for the trajectory network, residual calculation, individual loss terms, training loop, validation, checkpoint saving, and trajectory plots against the numerical reference. Add the losses incrementally and log each separately.

### Done when
The model trains reproducibly, initial conditions are satisfied to a measured tolerance, and trajectory error and equation residual are evaluated separately.

## Chapter 6.4 — Diagnose PINN training

Compare data-only training, physics plus initial conditions, and combined data/physics losses. Test a few loss weights and collocation-point distributions.

Measure held-out trajectory error, physics residual, initial-condition error, convergence, and runtime. A PINN is not automatically better; if it performs worse, investigate and report why.

### Done when
The result is reproducible and includes both successful and failed configurations.

---

# MILESTONE 7 — Compare Methods Fairly

Build one evaluation script for the numerical solver, simple baseline, conventional neural network, and PINN.

Use the same held-out cases and metrics. Do not tune on test data. Report failed/unresolved simulations, training and inference time separately, and errors in difficult regimes as well as overall metrics.

### Full-code deliverable
Load the saved models and baseline, evaluate identical cases, write a results table, and generate comparison plots.

### Done when
Results can be regenerated from documented commands and saved checkpoints.

---

# MILESTONE 8 — Choose One Research Extension

Only after the baseline comparison works, choose one:

- **Inverse parameter estimation:** infer ray parameters from trajectory/deflection observations.
- **Adaptive sampling:** spend more simulation budget where numerical or model error is high.
- **Critical-trajectory learning:** study near-critical scattering after careful solver validation.
- **Alternative PINN objectives:** investigate initial conditions, collocation sampling, and loss weighting.

Choose based on evidence, not by attempting everything at once.

---

# MILESTONE 9 — Publish the Project

Deliver:
- README with setup and quick start,
- this guide aligned with the actual implementation,
- tested environment/dependency instructions,
- commands to simulate, generate data, train, and evaluate,
- reproducible configurations,
- figures with units/captions,
- metrics and failure analysis,
- report covering assumptions, method, results, and limitations,
- a release tested in a clean environment.

The report must distinguish measured results from hypotheses. A failed hypothesis is acceptable; an irreproducible result is not.

---

# 5. The Very Next Session: Chapter 1.1

We will:
1. inspect the actual current geodesics.py and simulate.py,
2. identify import-time experiments and side effects,
3. refactor the files without changing the physics,
4. show the complete contents of every changed file,
5. explain the new code and design decisions,
6. run the simulation and an import-side-effect check,
7. verify the trajectory behavior is still sensible,
8. commit if the checks pass.

We will not jump to PINN implementation until the simulator can produce trustworthy, testable results.

---

# 6. Completion Checklist

- [ ] **M1:** clean simulator API, explicit status, deflection function, tests, convergence study.
- [ ] **M2:** weak-field validation, critical-regime investigation, reproducible plots.
- [ ] **M3:** documented dataset schema, generator, reproducible splits, leakage prevention.
- [ ] **M4:** baseline and fixed evaluation protocol.
- [ ] **M5:** conventional NN with training, checkpointing, evaluation, and error analysis.
- [ ] **M6:** automatic derivatives, PINN residual and conditions, diagnostics.
- [ ] **M7:** fair comparison of solver, baseline, NN, and PINN.
- [ ] **M8:** one research extension evaluated with evidence.
- [ ] **M9:** clean setup, reproducible outputs, report, and release.

**Use this file as the project syllabus.** Each chapter becomes a detailed build-along lesson when we reach it. The next concrete task is Chapter 1.1, not rewriting code you already understand.
