# Black Hole Lab — Model Arena

The Model Arena is a **precomputed presentation dashboard**. Training and model selection happen before the talk. Opening /arena, selecting a test ray, or switching visualizations never starts training.

## Approaches

| Approach | What it predicts | Evaluation |
|---|---|---|
| DOP853 | High-accuracy numerical reference | Reference only; not a learned model |
| MLP | Next (u, v) state from supervised examples | Autoregressive rollout |
| PINN | u(r0, phi) directly | Autograd derivative gives v = du/dphi; ODE residual and initial conditions form its loss |
| PPO | Next (u, v) state using teacher-guided reward | Deterministic autoregressive rollout |
| Adaptive RL | RK4 integration step size | Policy-controlled integration, sampled on the DOP853 test grid |

The PINN equation and initial conditions are

\[
u'' + u - \frac{3}{2}u^2 = 0,\qquad u(0)=\frac{1}{r_0},\qquad u'(0)=0.
\]

The PINN trainer uses training-ray closest-approach values and collocation points. It does not minimize trajectory-label error. Validation is based on the physics residual and initial conditions.

The adaptive RL environment has a different action space from the state-prediction PPO environment: its action selects the next RK4 step size. DOP853 interpolation is used to compute teacher-guided reward during training and to score evaluation, but future reference states are not observations given to the policy.

## Install

~~~powershell
python -m pip install -e ".[dev,ml,rl]"
python -m pytest -q
~~~

## Train saved models before the presentation

For a single preparation run, use the existing parallel pipeline to train the MLP and state-PPO, and select the PPO checkpoint using autoregressive validation error:

~~~powershell
python -m presentation.run_parallel --run-name class_presentation --rl-seconds 600 --mlp-epochs 300
python -m presentation.train_pinn --seconds 600 --output-dir outputs/pinn_runs/class_pinn_10m
python -m rl.train_adaptive --seconds 600 --run-name class_adaptive_10m
~~~

The selected state-PPO checkpoint is saved at outputs/presentation/class_presentation/selected_ppo_model.zip. The adaptive policy's best checkpoint is normally at outputs/adaptive_rl_runs/class_adaptive_10m/best/best_model.zip; if that file was not produced before the time budget ended, use final_model.zip.

~~~powershell
python -m presentation.evaluate_arena --mlp outputs/mlp_runs/dynamics_architecture_sweep_relu_w128_d3/best_model.pt --ppo outputs/presentation/class_presentation/selected_ppo_model.zip --pinn outputs/pinn_runs/class_pinn_10m/best_model.pt --adaptive-ppo outputs/adaptive_rl_runs/class_adaptive_10m/best/best_model.zip --output outputs/presentation/model_arena
~~~

If the adaptive PPO best_model.zip does not exist, substitute its final_model.zip. The evaluator writes predictions.csv, metrics_by_ray.csv, summary.csv, summary.json, run_metadata.json, and an overview plot.

Start the local results dashboard:

~~~powershell
python -m bh_analytics.dashboard --root outputs
~~~

Open http://127.0.0.1:8765/arena. It reads predictions.csv, metrics_by_ray.csv, summary.json, run_metadata.json, and (if generated) budget_projection.csv.

## Training-budget experiment

The full budget sweep trains each selected approach for 1 minute, 10 minutes, and 1 hour, then evaluates the resulting checkpoints on the same validation rays:

~~~powershell
python -m presentation.run_budget_sweep
~~~

This default command takes roughly 4 hours 44 minutes of requested training time for four methods (plus evaluation and any epoch/callback overhead). It is an offline preparation task, not a live presentation feature. To run only one budget, use --budgets 60. The sweep requires MLP, PINN, and PPO; adaptive RL is optional and can be omitted with --models MLP PINN PPO.

The script writes measured observations to budget_measurements.csv. The 10-hour and 10-day entries in budget_projection.csv are generated only when at least two measured checkpoints exist for a model. They are log-log extrapolations, clearly marked projected, and are not training runs or empirical results. If there is insufficient evidence, the projection is left blank.

## Fairness and limitations

- MLP, PINN, state-PPO, and adaptive RL are scored against the same held-out complete test rays and DOP853 reference.
- MLP/PPO receive only their own previous predicted state during rollout.
- PINN uses its learned function of (r0, phi) and autograd for its derivative; this is not the same rollout mechanism, so the report states this distinction.
- Adaptive RL chooses RK4 step sizes; accuracy is measured after interpolation onto the reference angle grid. Report both error and number of RK4 steps/function evaluations.
- Training budgets must be actual wall-clock durations recorded in run configuration. Do not replace missing results with guessed numbers.
- DOP853 is a high-accuracy numerical reference, not an analytical exact solution.
- The current tracked dataset has only 30 trajectories and a five-ray test split. Treat results as a small controlled prototype, not broad proof of generalization.
