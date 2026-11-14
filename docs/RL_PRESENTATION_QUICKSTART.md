# Black Hole Lab — RL presentation prototype

## Research scope

This is a **teacher-guided sequential trajectory-tracking experiment**. PPO
observes only the initial radius, current angle, and its own current predicted
state. The DOP853 next state is used to calculate reward and evaluation
metrics; it is not included in the policy observation.

This prototype does not claim that the agent independently discovers general
relativity, solves the geodesic ODE from first principles, or learns physical
capture versus escape. Episodes end at the final sample of the reference ray.

## Install

~~~powershell
python -m pip install -e ".[dev,rl]"
~~~

## Train

~~~powershell
python -m rl.train --seconds 60 --seed 42 --run-name 1min
python -m rl.train --seconds 600 --seed 42 --run-name 10min
python -m rl.train --seconds 3600 --seed 42 --run-name 1hour
~~~

Runs save config, initial/final models, checkpoints, TensorBoard logs, and validation
histories under outputs/rl_runs/<run-name>/. To watch training live in a second terminal:

~~~powershell
tensorboard --logdir outputs/rl_runs
~~~

Then open the local TensorBoard address printed in the terminal (usually port 6006).

## Evaluate and compare

First evaluate the untrained policy as a baseline, then the trained policy:

~~~powershell
python -m rl.evaluate --model outputs/rl_runs/1min/initial_model.zip --split test --output outputs/rl_runs/1min/evaluation_initial
python -m rl.evaluate --model outputs/rl_runs/1min/final_model.zip --split test --output outputs/rl_runs/1min/evaluation_final
python -m rl.evaluate --model outputs/rl_runs/1min/best/best_model.zip --split test
python -m rl.evaluate --model outputs/rl_runs/10min/best/best_model.zip --split test
python -m rl.compare outputs/rl_runs/1min outputs/rl_runs/10min outputs/rl_runs/1hour
~~~

If best/best_model.zip does not exist because the run ended before its first
evaluation, use final_model.zip.

## Observation, action, and reward

Observation: [r0/12, phi/3, current_u/0.5, current_v/0.2].

Action: two normalized continuous values mapped to u in [0, 0.5] and v in
[-0.2, 0.2].

For proposed state (u_p, v_p) and DOP853 target (u*, v*):

~~~text
e_u = (u_p - u*) / 0.5
e_v = (v_p - v*) / 0.2
reward = 1 - 4 * (e_u^2 + e_v^2)
~~~

Reward near 1 means the proposed state is close to the reference. The teacher
target is available to reward/evaluation code, not the policy observation.

## Presentation caveats

- PPO may not improve monotonically.
- A 1-minute run may only prove the pipeline executes, not convergence.
- Compare all models on the same held-out test rays.
- Include DOP853 as the numerical reference baseline.
- Report actual wall-clock time, timesteps, seed, and measured errors.
- If PPO does not improve, report that honestly.
- The numerical solver is expected to remain more accurate on the trajectories
  it directly integrated.
