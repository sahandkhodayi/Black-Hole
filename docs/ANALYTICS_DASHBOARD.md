# Interactive Black Hole Lab classroom

This local website is built to demonstrate the differential equation visually. Start with the photon-orbit explorer, then compare the numerical solution with the PPO agent's prediction.

## Open the website

From the Black Hole Lab project folder, in PowerShell:

```powershell
uv pip install -e ".[rl]"
python -m bh_analytics.dashboard --root outputs
```

Open `http://127.0.0.1:8765` if the browser does not open automatically. Keep the terminal running while using the page.

For WSL:

```bash
python -m bh_analytics.dashboard --root outputs --no-browser
```

Then open `http://127.0.0.1:8765` in the Windows browser.

## 1. Explore the differential equation

The first panel runs the actual DOP853 solver. Move the **closest approach** slider and release it to trace a new path automatically. You can also press **Trace this path**. You can play the photon moving along the orbit; the two state plots follow the same position.

The model uses normalized units with Schwarzschild radius `r_s = 1` and defines inverse radius `u = 1/r`. Its orbit equation is:

```text
d²u/dφ² + u = 3/2 u²
```

To integrate a second-order equation as a first-order system, define `v = du/dφ`:

```text
u' = v
v' = 3/2 u² - u
```

At closest approach `r₀`, the initial state is `u(0) = 1/r₀`, `v(0) = 0`. The solver advances that state in the angle `φ`.

The visual shows:
- the event horizon at `r/r_s = 1`;
- the photon sphere at `r/r_s = 1.5`;
- the photon orbit calculated by DOP853;
- `u(φ)`, the inverse radius;
- `v(φ)`, its derivative.

The deflection angle shown beside the simulation is calculated from the solver's escape angle. The asymptotic ends of the path are clipped to the edge of the plot for display; the solver values `u` and `v` are not clipped.

## 2. Compare the AI prediction

The second panel can start a PPO run. The model predicts the next `(u, v)` state, and its own prediction becomes the input for the next step. The reference curve is generated from DOP853.

When the trainer completes a validation rollout, the website draws:
- **Solid cyan:** DOP853 reference trajectory.
- **Dashed pink:** the PPO model's autoregressive prediction.

A gap between the curves is the prediction error. The panel also displays validation RMSE for (u) and (v), and a small reward history. Reward is a training signal; the orbit overlay and validation RMSE are more directly useful for judging how well the policy follows the trajectory.

Choose a run length and seed, then press **Start training**. Press **Stop & save** to request a safe stop. The trainer saves its current model and the best validation checkpoint.

The browser requests a new training snapshot **once every 60 seconds**. It does not poll every second. The first PPO orbit appears after the first validation rollout. Validation itself is also run at the snapshot interval.

The browser controls currently start PPO only. The supervised MLP and paired MLP-vs-PPO classroom training commands remain available from the command line.

## 3. Saved experiments

The final section lets you inspect existing MLP, PPO, and paired comparison runs, open their saved figures, and save a standalone HTML report. The overall result is read from the saved summary; you do not need to create or move CSV files manually.

## Notes for a class presentation

This project does **not** train an AI to discover Einstein's equations. The Schwarzschild orbit equation is supplied by the physics model. DOP853 numerically integrates it. PPO is an experimental, teacher-guided method for approximating the trajectory step by step.

The numerical solver is the reference. The AI curve is an approximation and may visibly diverge, especially because each predicted state is fed into the next prediction. Showing that difference is part of the experiment, not a rendering error.

## Local-only safety

Browser training is enabled only when the dashboard is bound to localhost. The start/stop endpoints are intended for the user's own machine, not for an exposed multi-user server.
