# Black Hole Lab — RL Learning Guidebook

**The active manual from the current repository state onward**

> **Working principle:** I will help you understand and build the project, not silently build the research contribution for you. You write the important RL decisions and implementation. I explain the surrounding theory, give hints in stages, review your code, help debug it, and only reveal a full solution when you have made a serious attempt or explicitly ask for one.

This book starts from the working physics and dataset foundation already in the repository. It is not a Python beginner course, and it is not a copy-paste tutorial. It is a sequence of research milestones.

---

# How we will work together

## The no-cheating agreement

For the physics and data-engineering foundation, complete reference implementations are acceptable because those parts are infrastructure and the repository already contains them.

For the RL contribution, we will use a different teaching process:

1. I explain the concept and why the project needs it.
2. I give you the interface, constraints, and a small task.
3. You write the first implementation.
4. You run it and share the code, output, or traceback.
5. I review it, identify the actual issue, and give a hint.
6. Hints become more specific only when needed.
7. You make the correction and explain why it works.
8. We record the decision and only then move to the next milestone.

I will not immediately paste a finished environment, reward, PPO training pipeline, or agent. Those are the parts you are supposed to learn by implementing.

### The hint ladder

When you are stuck, ask for one of these:

| Level | What I give you |
|---|---|
| Hint 1 | A conceptual nudge; no code |
| Hint 2 | The relevant equation, API, or data-flow clue |
| Hint 3 | Pseudocode or a partial code skeleton |
| Hint 4 | A localized code example for the specific difficult block |
| Full explanation | A complete worked solution, only after an attempt or when you explicitly request it |

You can say: "Give me hint 1", "review this without solving it", or "show me the smallest code example."

### What counts as your implementation?

You should be able to answer these questions about every important function:

- What enters it?
- What comes out?
- What state does it change?
- Which mathematical or RL concept does it implement?
- What could go wrong?
- How did you verify it?

If you can only say "the code works because ChatGPT gave it to me", we are not done with that milestone.

## What I will and will not do

I will:
- explain unfamiliar mathematics and APIs;
- provide small code examples for concepts;
- review your actual code;
- help interpret errors and test failures;
- challenge assumptions in the experiment;
- suggest experiments and diagnostics;
- help write the report based on results you actually obtain.

I will not:
- claim your experiment succeeded without evidence;
- invent training results;
- quietly change the research question;
- tune the reward just to make a plot look good;
- call a teacher-guided experiment "the agent discovering general relativity";
- dump the entire RL solution before you have worked through the design.

---

# Part I — The project and its scientific question

## 1. What are we trying to build?

The research question is:

**Can an agent learn to generate a photon trajectory near a Schwarzschild black hole, reproduce gravitational light bending, and generalize to initial conditions it has not encountered during training?**

The project has two distinct systems:

1. A numerical physics system that calculates trusted reference trajectories.
2. A learning system that attempts to generate trajectories and is evaluated against those references.

The numerical solver is not the enemy of the learning system. It is our reference instrument. We need to know what a correct trajectory looks like before we can judge a generated one.

## 2. What the project is not

- It is not an RL agent choosing the ODE solver's step size.
- It is not a chatbot or a RAG project.
- It is not just a model that predicts one deflection number.
- It is not a claim that a neural network has rediscovered Einstein's equations.
- It is not a full general-relativity simulator. The current model is a restricted Schwarzschild photon-orbit equation with normalized units and a scattering-ray setup.

The first version studies **escaping/scattering rays**. Captured rays need a different initial-value setup and a separate capture event; do not mix them into this first dataset.

## 3. What the current code already does

| File | Responsibility |
|---|---|
| src/lab/geodesics.py | ODE, event function, DOP853 simulation, mirrored trajectory object |
| src/lab/reference.py | Dense-output DOP853 reference object |
| src/lab/trajectory.py | Samples and packs reference trajectories |
| src/lab/splits.py | Reproducible whole-ray split logic |
| src/generate_data.py | Dataset generation command |
| src/inspect_data.py | Dataset validation and trajectory gallery |
| src/make_splits.py | Split-manifest command |
| src/lab/integrators.py | Classical RK4 one-step integrator |
| src/lab/plot.py | Photon trajectory plots |
| src/lab/compare_plot.py | RK4/DOP853 comparison prototype |
| tests/ | Physics, reference, dataset, inspector, and split tests |

The repository has a working pre-learning foundation. Your local run of the test suite is the evidence that this foundation passed in your environment. Keep running the suite after changes.

---

# Part II — Physics you must understand before designing an agent

## 4. Units and coordinates

We normalize the Schwarzschild radius:

\[
r_s=1.
\]

All radii in the simulation are therefore measured in Schwarzschild-radius units.

The independent variable is the polar angle:

\[
\phi\quad\text{(radians)}.
\]

It is not physical time.

We define inverse radius:

\[
u=\frac{1}{r}
\]

and its derivative with respect to angle:

\[
v=\frac{du}{d\phi}.
\]

The state is ordered as:

\[
s(\phi)=
\begin{bmatrix}
u(\phi)\\
v(\phi)
\end{bmatrix}.
\]

Do not swap u and v in arrays. The repository's solver expects [u, v].

## 5. The differential equation

The existing photon-orbit equation is:

\[
\frac{d^2u}{d\phi^2}=\frac{3}{2}u^2-u.
\]

With v = du/dphi:

\[
\boxed{
\begin{aligned}
u'&=v,\\
v'&=\frac{3}{2}u^2-u.
\end{aligned}}
\]

The function src/lab/geodesics.py implements the right-hand side.

At closest approach r0:

\[
u_0=\frac{1}{r_0},\qquad v_0=0,\qquad\phi_0=0.
\]

The first dataset uses r0 greater than 1.5. In these units, 1.5 is the photon-sphere radius. The event horizon is at radius 1.

## 6. Escape and deflection

Because u = 1/r, an escaping ray approaches:

\[
r\to\infty\quad\Longleftrightarrow\quad u\to0.
\]

The event function detects an outward crossing of u=0. A negative crossing direction is used because u decreases as an escaping photon moves outward.

The escape angle is the angular coordinate at which that event is detected:

\[
\phi_{\mathrm{escape}}.
\]

For the symmetric scattering setup, the total deflection is:

\[
\boxed{\alpha=2\phi_{\mathrm{escape}}-\pi}.
\]

Escape angle and deflection angle are not the same measurement. The former is the endpoint's angular coordinate; the latter is the change from a straight, undeflected path.

## 7. A useful invariant

The ODE has a conserved quantity:

\[
I(u,v)=v^2-u^3+u^2.
\]

At closest approach, v0=0 and u0=1/r0, so:

\[
I_0=u_0^2-u_0^3.
\]

For an exact trajectory, this value stays constant. A numerical solver introduces a small amount of drift. We can use invariant drift as one diagnostic of physical consistency.

Important: preserving this invariant is necessary for a physically consistent solution of this ODE, but it is not by itself sufficient to prove that a generated trajectory is correct. We also compare the curve to the reference and check the ODE residual.

## 8. Check your understanding

Do not skip this. Answer these in your own words before designing the environment:

1. Why is phi not physical time in this formulation?
2. If u decreases, what happens to r?
3. Why does v0 equal zero at closest approach?
4. What does the event u=0 represent?
5. Why does a larger escape angle imply more deflection in this symmetric scattering setup?
6. What does invariant drift tell us, and what can it not prove?

**Checkpoint:** You can explain the state [u, v], initial conditions, escape event, escape angle, deflection, and invariant without reading the definitions.

---

# Part III — Generate and understand the reference data

## 9. Why do we need a reference dataset?

A learning system needs examples and a way to judge its outputs. The DOP853 solver gives us high-accuracy numerical reference trajectories for selected closest approaches.

This does not mean we feed the future part of a reference trajectory into the agent. The reference is stored for evaluation and, depending on the RL formulation, may contribute to a training reward. The policy observation must be designed separately.

## 10. Dense output: adaptive solver points versus dataset points

SciPy's solve_ivp chooses adaptive internal steps. The accepted angular points are stored in solution.t.

When dense_output=True, solution.sol is an interpolating function. It estimates the state at requested angles inside the integrated interval.

That lets us sample the reference on a chosen grid:

\[
\phi_k=k\Delta\phi
\]

and include the exact escape-event angle as the final sample.

The last interval can be shorter than delta_phi. That is intentional.

## 11. Understand the packed NPZ format

The dataset stores variable-length trajectories in flat arrays.

| Array | Meaning |
|---|---|
| case_id | Original identifier for the requested ray |
| r0 | Closest-approach radius for each successfully escaped ray |
| phi | All angular samples concatenated together |
| u | Corresponding inverse-radius samples |
| v | Corresponding derivative samples |
| offsets | Start/end positions for each ray in the packed arrays |
| escape_angle | Escape angle for each ray |
| deflection_radian | Total deflection for each ray |

For ray i, its samples are:

\[
[\mathrm{offsets}[i],\ \mathrm{offsets}[i+1]).
\]

Example: if offsets are [0, 100, 205, 310], then:
- ray 0 uses samples 0 through 99;
- ray 1 uses samples 100 through 204;
- ray 2 uses samples 205 through 309.

The end index is excluded, like a normal Python slice.

## 12. Your data-preparation task

Do not overwrite the tracked example dataset. Use a new filename:

~~~powershell
python src/generate_data.py --count 30 --r-min 2.2 --r-max 12 --output data/photon_trajectories_v2.npz --manifest data/trajectory_manifest_v2.json
~~~

Inspect it:

~~~powershell
python src/inspect_data.py --data data/photon_trajectories_v2.npz --manifest data/trajectory_manifest_v2.json --show
~~~

Create a split manifest:

~~~powershell
python src/make_splits.py --data data/photon_trajectories_v2.npz --output data/split_manifest_v2.json --seed 42
~~~

Run the tests:

~~~powershell
python -m pytest -q
~~~

## 13. Inspect the actual arrays yourself

Open a Python shell from the project root:

~~~python
import numpy as np

data = np.load("data/photon_trajectories_v2.npz", allow_pickle=False)

print(data.files)
print("number of rays:", len(data["r0"]))
print("number of packed phi samples:", len(data["phi"]))
print("offsets:", data["offsets"][:5])

ray_index = 0
start = data["offsets"][ray_index]
end = data["offsets"][ray_index + 1]

phi = data["phi"][start:end]
u = data["u"][start:end]
v = data["v"][start:end]

print("r0:", data["r0"][ray_index])
print("first state:", u[0], v[0])
print("last state:", u[-1], v[-1])
print("last angle:", phi[-1])
print("stored escape angle:", data["escape_angle"][ray_index])
~~~

### Your exercise

Without asking me to write the code:
- calculate the radius at the first sample;
- calculate the maximum radius before the escape endpoint;
- verify that phi increases;
- verify that the final u is close to zero;
- compare the calculated deflection against the stored value.

**Hints:**
- Radius is the reciprocal of positive u.
- The last sample is at the escape angle.
- Use numpy's diff, all, max, and isclose functions.
- Do not attempt 1/u at u=0 without handling division safely.

## 14. What a good reference dataset looks like

For every stored ray:
- phi starts at zero;
- u starts at 1/r0;
- v starts at zero;
- phi is strictly increasing;
- u, v, and phi are finite;
- the final angular sample equals the stored escape angle;
- the final u is numerically close to zero;
- the stored deflection agrees with 2*escape_angle-pi.

The inspector checks much of this. You should still learn to inspect one ray manually. A green test suite does not replace understanding the arrays.

---

# Part IV — Splitting data without leakage

## 15. Why split by ray, not by point?

Neighboring points on the same trajectory are extremely correlated. If some points from one ray go into training and adjacent points from that same ray go into test, the evaluation is misleading.

The independent case is the entire ray, identified by case_id.

The split utility therefore partitions case IDs. It never partitions individual phi samples.

Default proportions are 70% train, 15% validation, and 15% test. With a small dataset, integer rounding means the actual counts may differ slightly from those percentages.

## 16. What each split is for

- **Training:** may be used to fit policy parameters and training-only normalization statistics.
- **Validation:** used to compare design choices and select checkpoints.
- **Test:** held back for final evaluation. Do not repeatedly tune against it.

Never fit normalization ranges, means, or standard deviations using validation or test cases if the result is meant to represent held-out generalization.

## 17. What kind of generalization does this split measure?

A random ray-level split from a continuous r0 interval mostly tests interpolation among unseen rays from the same range.

It does not automatically establish extrapolation beyond the range. Later, create a separate challenge set with r0 values outside the training interval or closer to the photon sphere. Keep that challenge set separate from ordinary validation.

**Checkpoint:** Your split manifest is saved, deterministic for a fixed seed, has no overlapping case IDs, and includes every successfully stored ray exactly once.

---

# Part V — Reinforcement learning concepts, applied to this problem

## 18. The RL loop

A standard RL interaction has:

- **Agent:** the policy that chooses actions.
- **Environment:** the system that receives actions and returns observations, rewards, and episode status.
- **Observation:** the information the policy is allowed to see now.
- **Action:** what the policy proposes.
- **Transition:** how the environment moves to the next state.
- **Reward:** a scalar signal describing how desirable the action/outcome was.
- **Episode:** one complete interaction sequence.

In our proposed task, one episode represents one photon ray.

The goal is not to output a single number. It is to generate an ordered sequence of states that forms a curve.

## 19. The crucial design question: what does the agent control?

The current project direction is:

**The agent proposes the next photon state. It does not choose the angular step size.**

The environment uses a fixed angular increment:

\[
\Delta\phi=\text{constant}.
\]

At a normal step:

\[
\phi_{t+1}=\phi_t+\Delta\phi.
\]

The agent proposes a next state:

\[
(u_{t+1},v_{t+1}).
\]

This is different from asking the agent to choose the next integration step. The angle increment is fixed by the environment.

## 20. A scientific honesty warning: teacher-guided RL

The reference solver gives us the correct future state. If the environment compares the agent's proposal against that future reference state and rewards small error, then the reward contains teacher information.

The policy does not directly observe the reference target, but it is being guided by it through reward.

That is a valid experimental design if described accurately. It is **teacher-guided trajectory generation through RL**. It is not the agent discovering the Schwarzschild equation from scratch.

There is also a research risk: because the reference is deterministic and directly provides the desired next state, ordinary supervised learning may solve the mapping more simply. We will not pretend RL is automatically the best tool. The scientific result includes whether the RL formulation is useful compared with a simpler baseline later.

## 21. Two possible episode-ending designs

Resolve this before writing the environment. These designs answer slightly different questions.

### Design A — Reference-bounded rollout (recommended first prototype)

The environment knows the reference escape angle and ends the episode at that point. At each step, the policy proposes the next state. The reference future state is used only for reward/evaluation, never included in the policy observation.

Advantages:
- easier first environment;
- every episode has a known reference horizon;
- lets you focus on sequential state generation and reward design.

Limitation:
- the environment, not the policy, decides when the ray ends;
- the policy is not learning autonomous escape detection;
- if you include normalized progress based on the reference escape angle in the observation, you reveal reference horizon information.

For the first prototype, do not put future reference u, future reference v, escape angle, or target remaining steps in the observation. Give the policy current state and the initial ray parameter. The environment may still use the stored endpoint internally to score and stop the teacher-bounded rollout.

### Design B — Agent decides when to stop (later, harder)

The policy proposes the next state and also emits a stop/continue decision. The environment has a generous maximum-step safety limit, but the policy must learn when it is appropriate to stop.

Advantages:
- closer to autonomous trajectory generation;
- the agent learns an explicit episode-ending decision.

Limitations:
- adds another action dimension and another reward design problem;
- premature stopping and late stopping need carefully designed penalties;
- reference endpoint must remain hidden from the observation.

Do not implement both designs at once. Start with A, understand it, and only then decide whether B is worth the additional complexity.

## 22. Observation design

The observation is not "everything the environment knows". It is only what the policy is allowed to know at the current step.

A reasonable first design discussion includes:

- initial closest approach r0;
- current angular coordinate phi;
- current inverse radius u;
- current derivative v.

You must decide whether to normalize them and how. The observation must not include the future reference state.

### Your task

Write down the exact observation vector and explain every element. For every component answer:

1. Is it available at the current step?
2. Is it available without querying the teacher's future trajectory?
3. Is it bounded or can it vary significantly?
4. How will it be scaled?
5. Could it accidentally reveal the reference endpoint?

Do not code the observation until you can answer those questions.

## 23. Action design

For Design A, the initial action can be a two-component continuous action representing the next proposed normalized u and v.

Conceptually:

\[
a_t=(\tilde u_{t+1},\tilde v_{t+1}).
\]

The environment decodes the normalized action into physical u and v units.

Questions you must answer:
- What are the action bounds?
- Are they derived using training cases only?
- What happens if a proposed u is negative?
- What happens if the action contains NaN or infinity?
- Does clipping hide a policy failure?
- Are you predicting absolute next state or a correction to the current state?

For a first version, absolute next-state prediction is easier to reason about. Later you can experiment with residual actions, but do not change representation while debugging the first environment.

**Important:** A bounded action space does not guarantee a physically correct action. Bounds only constrain the output range.

## 24. Transition logic

The environment, not the agent, advances phi by the fixed angular amount.

A normal step conceptually does this:

1. Read current ray and current state.
2. Receive the agent's proposed next state.
3. Validate the proposal.
4. Advance phi by the fixed angular increment.
5. Compare the proposal with the reference at the new phi for reward/evaluation.
6. Calculate diagnostic residuals.
7. Decide whether the episode terminated or hit a safety limit.
8. Return next observation, reward, termination flags, and diagnostic information.

Do not copy this list into code without thinking. It is the contract your implementation must satisfy.

## 25. Gymnasium interface you will implement

Gymnasium environments typically provide:

~~~python
reset(seed=None, options=None)
step(action)
~~~

Your environment will eventually expose:

- observation_space;
- action_space;
- reset;
- step;
- close (if it owns any external resources).

The exact method contract and return values must follow the Gymnasium version you install. Read the official API rather than relying on memory.

Documentation:
https://gymnasium.farama.org/api/env/

### Do not start by writing the whole environment

First implement a tiny object that can select one stored ray and reset to its initial state. Test the data flow manually.

Then add one fixed angular advance without actions.

Then add an action placeholder and check shape/bounds.

Then add one diagnostic.

Then add reward.

Then add episode-end handling.

This sequence makes bugs local and understandable.

---

# Part VI — Reward engineering

## 26. Why reward is difficult

The reward is the definition of success as seen by the optimizer. If the reward is incomplete, the agent can learn a behavior that scores well but is physically wrong.

A curve can:
- look visually plausible but have the wrong escape angle;
- match u while having an incorrect v;
- preserve the invariant but follow the wrong branch;
- track the teacher early and drift later;
- exploit clipping or endpoint logic.

Therefore, one scalar reward is not enough for scientific evaluation. Save its components separately.

## 27. Three candidate signal families

These are design ingredients, not a finished reward formula.

### A. Reference-tracking error

At the next angle, compare the proposed state with the DOP853 reference:

\[
e_u=u_{\mathrm{agent}}-u_{\mathrm{ref}},
\qquad
e_v=v_{\mathrm{agent}}-v_{\mathrm{ref}}.
\]

Normalize these errors by scales computed from training rays only.

A reward component may penalize the squared normalized errors.

**Question:** Should u and v have equal weights? Think about their scales and physical roles before deciding.

### B. Discrete ODE consistency

The continuous system says:

\[
u'=v,\qquad v'=f(u),\qquad f(u)=\frac32u^2-u.
\]

A simple midpoint-style consistency check over one fixed interval is:

\[
R_u=
u_{t+1}-u_t
-\frac{\Delta\phi}{2}(v_t+v_{t+1}),
\]

\[
R_v=
v_{t+1}-v_t
-\frac{\Delta\phi}{2}
\left(f(u_t)+f(u_{t+1})\right).
\]

These residuals are approximate discrete consistency errors, not exact identities for a finite step.

**Question:** What happens to these residuals when delta_phi becomes smaller? What scale should you use before comparing the two residuals?

### C. Invariant drift

For each proposed state:

\[
I_t=v_t^2-u_t^3+u_t^2.
\]

Compare it with the initial invariant:

\[
e_I=I_t-I_0.
\]

This can penalize drift from the physical invariant.

Remember: the invariant is a diagnostic constraint. It does not uniquely identify the correct point on the orbit.

## 28. Reward design exercise

Before coding, write a table like this:

| Component | What it measures | Why it belongs | Normalization | Failure it catches |
|---|---|---|---|---|
| Reference tracking | You fill this | You fill this | You fill this | You fill this |
| ODE consistency | You fill this | You fill this | You fill this | You fill this |
| Invariant drift | You fill this | You fill this | You fill this | You fill this |
| Invalid action | You fill this | You fill this | You fill this | You fill this |
| Terminal behavior | You fill this | You fill this | You fill this | You fill this |

Then choose initial weights and justify them. Do not tune against the test set.

### Hints

- Normalize u and v errors before combining them.
- A reward term with a numerical magnitude 1000 times larger than another can dominate regardless of your intended priorities.
- Log each reward component separately.
- Begin with a simple reward and add one term at a time.
- Record why every term exists.
- Compare behavior before and after adding a term. Do not add five terms at once.

## 29. Reward exploitation checks

Construct intentionally bad proposals and ask:

- Can the agent get reward by outputting the same state every step?
- Can it produce negative u and still be rewarded?
- Can it terminate immediately?
- Can it continue after the reference endpoint?
- Can it satisfy the invariant while having the wrong trajectory?
- Can it exploit action clipping?
- Can it maximize reward but have a poor escape angle?

If any answer is yes, revise the environment or reward before long training.

---

# Part VII — Building the environment by yourself

## 30. Environment implementation milestones

Implement these in order. Do not combine all tasks in one large edit.

### E0 — Load one ray

Your environment should receive a dataset and a list of permitted case IDs.

Task:
- map case IDs to packed-array positions;
- use offsets to extract one ray;
- return the initial physical state.

Hint: case_id is not guaranteed to equal the array position in all dataset versions. Do not assume it does.

Acceptance:
- selecting a case gives its correct r0;
- its first phi is zero;
- initial u is 1/r0;
- initial v is zero.

### E1 — Reset behavior

Task:
- reset internal step counter;
- select a case;
- restore phi, u, v;
- create the initial observation;
- return the correct Gymnasium reset output.

Hint: deterministic case selection is useful for debugging. Random selection is useful during training. Support both deliberately.

Acceptance:
- resetting the same fixed case twice returns the same initial state and observation;
- the environment does not accidentally select validation/test cases when constructed with training IDs.

### E2 — Fixed angular transition

Task:
- define delta_phi once;
- increment phi by exactly delta_phi for normal steps;
- handle the final partial interval if using reference-bounded rollout.

Hint: the final interval may be shorter than delta_phi. Do not step past the reference endpoint and then pretend the last point was exact.

Acceptance:
- every normal transition advances by the intended amount;
- endpoint logic is explicit and tested.

### E3 — Action validation

Task:
- validate action shape;
- validate finite values;
- map normalized actions back into physical state values;
- define what happens when the action is invalid.

Hint: action_space describes the contract. It is not a substitute for checking data at the boundary.

Acceptance:
- valid actions produce correctly shaped states;
- NaN, infinity, and wrong shapes fail safely;
- invalid proposals cannot receive successful terminal status.

### E4 — Observation

Task:
- implement the chosen observation vector;
- apply training-only scaling;
- return a finite array with the exact declared shape and dtype.

Hint: write the observation definition in a comment and keep one function responsible for creating it.

Acceptance:
- shape and dtype never change;
- no future reference u/v appears in the observation;
- evaluation uses the training scale values, not newly computed values.

### E5 — Reward components

Task:
- calculate and log reference error;
- calculate and log discrete ODE residual;
- calculate and log invariant drift;
- combine the components only after each has been checked independently.

Hint: a reward function should be easy to inspect. Prefer a small function that returns both total reward and a dictionary of named components.

Acceptance:
- all components are finite for valid actions;
- every component has a clear sign and interpretation;
- deliberately bad proposals receive worse signals than reasonable ones.

### E6 — Termination and truncation

Gymnasium distinguishes:
- terminated: the task's terminal condition has been reached;
- truncated: an external limit ended the episode, such as a safety horizon.

For reference-bounded Design A, termination is tied to the reference endpoint. Hitting a maximum-step safety limit before that should be truncation, not successful escape.

Acceptance:
- an unresolved ray is not mislabeled as escaped;
- an invalid proposal cannot produce successful termination;
- a safety limit is not confused with physical escape;
- step cannot silently continue after the episode has ended.

### E7 — Diagnostics

Return useful information for debugging:
- case_id;
- current phi;
- proposed u and v;
- reference u and v (in info for development/evaluation, never in observation);
- tracking errors;
- invariant drift;
- ODE residual components;
- termination reason.

Do not expose teacher targets through observation simply because they are convenient to log in info. Keep those channels separate.

---

# Part VIII — Prove the environment is understandable before training

## 31. A manual environment walkthrough

Before adding PPO, create a small script that:

1. creates the environment;
2. resets one known ray;
3. sends a hand-written sequence of actions;
4. prints the returned observation and reward components;
5. plots the resulting proposed curve against the reference;
6. closes the environment.

This is not a training run. It is a way to understand the transition and reward.

### Your first manual policies

Try three deliberately simple policies:

1. **Hold-state policy:** always proposes the current state.
2. **Reference-copy diagnostic:** uses the next reference state as a temporary debugging action.
3. **Random policy:** samples valid actions from the declared action space.

The reference-copy diagnostic is an oracle upper-bound/debugging tool. It is not an agent and must never be reported as learned performance. The hold-state and random policies are sanity baselines to reveal reward and termination bugs.

Questions:
- Does the reference-copy action receive better tracking reward?
- Does the hold-state action drift away?
- Does the random action fail safely?
- Do the reward components explain the total reward?

Do not start long PPO training until you can answer these questions.

## 32. Smoke checks versus the full test suite

The repository already has physics/data tests. As you add the environment, perform small smoke checks immediately. Do not postpone fixing basic breakage.

Later, expand the formal test suite for:
- reset determinism;
- observation shape and dtype;
- action shape and bounds;
- fixed phi increment;
- no teacher leakage;
- reward component finiteness;
- termination versus truncation;
- no steps after episode completion.

A five-second smoke run is not a substitute for unit tests, and a unit test is not a substitute for looking at the generated curve.

---

# Part IX — Only after the environment works: PPO

## 33. What is a policy?

A policy maps an observation to an action:

\[
a_t\sim\pi_\theta(a\mid o_t).
\]

The parameters theta are adjusted during training.

For continuous actions, a common policy represents a probability distribution over action values rather than directly returning one fixed number during training.

## 34. Return and value

A reward at one step is not the entire objective. The agent considers accumulated future reward.

A discounted return is:

\[
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
\]

where gamma controls how strongly later rewards matter.

The value function estimates expected future return from an observation:

\[
V_\phi(o_t)\approx\mathbb E[G_t\mid o_t].
\]

The advantage measures whether an action did better or worse than expected:

\[
A_t=Q(o_t,a_t)-V(o_t).
\]

You do not need to derive every PPO equation before the first experiment, but you must understand what the policy, value estimate, return, and advantage are doing.

## 35. Why PPO?

PPO is a practical first algorithm for continuous-control tasks. It constrains policy updates using a clipped objective so an update does not change the new policy ratio too aggressively.

A common clipped objective is:

\[
L^{\mathrm{CLIP}}(\theta)=
\mathbb E_t\left[
\min\left(
\rho_t(\theta)A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right],
\]

where:

\[
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid o_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)}.
\]

The clip does not guarantee good learning. PPO can still fail because of poor observations, reward scale, environment bugs, or insufficient exploration.

Read:
- PPO paper: https://arxiv.org/abs/1707.06347
- Stable-Baselines3 PPO documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- Gymnasium API: https://gymnasium.farama.org/api/env/

## 36. Your PPO learning tasks

Do not copy hyperparameters from a random tutorial without understanding them.

You will choose and justify:
- policy network size;
- learning rate;
- rollout length;
- batch size;
- number of epochs per update;
- discount factor;
- GAE lambda;
- entropy coefficient;
- clipping range;
- total timesteps;
- random seed.

For each parameter write:
1. what it controls;
2. what happens if it is too large;
3. what happens if it is too small;
4. what metric or behavior would suggest changing it.

Start with a small smoke run. A smoke run checks that interaction, rollout collection, logging, saving, and loading work. It is not evidence that the policy learned the physics.

Install RL dependencies only when you reach this part:

~~~powershell
python -m pip install gymnasium stable-baselines3 tensorboard
~~~

Keep the physics/data dependencies separate from the RL-specific dependencies if you want the base simulator to remain lightweight.

---

# Part X — Evaluation: a beautiful curve is not enough

## 37. Compare on a common angular grid

The agent and reference may have different arrays or episode lengths. Compare them on a common valid phi grid, not by blindly subtracting array indices.

For a common interval:

\[
E_u=\max_\phi|u_{\mathrm{agent}}-u_{\mathrm{ref}}|,
\]

\[
E_v=\max_\phi|v_{\mathrm{agent}}-v_{\mathrm{ref}}|.
\]

Also calculate:
- mean absolute u error;
- mean absolute v error;
- escape-angle error;
- deflection-angle error;
- mean Cartesian curve distance over a clearly defined finite plotting interval;
- maximum invariant drift;
- maximum discrete ODE residual;
- valid escape rate;
- episode length and runtime.

The Cartesian metric needs care because radius diverges at the asymptotic endpoint. Define the finite plotting/range mask explicitly and use the same mask for both curves.

## 38. Held-out evaluation

Use the saved split manifest. Do not regenerate random splits every time.

Report separately:
- training performance;
- validation performance;
- final test performance.

Only use validation results to make design choices. Once the design is frozen, evaluate the test set.

The random split tests unseen trajectories within the sampled range. Add a separate extrapolation or near-critical challenge set later.

## 39. Multiple random seeds

One successful training seed is not strong evidence. After the first pipeline works, train several seeds and report:
- mean performance;
- standard deviation or a suitable spread;
- failure rate;
- representative successful and failed curves.

Do not show only the best run.

---

# Part XI — Scientific comparisons after the RL version works

These are later experiments. They do not belong in the first implementation sprint.

## 40. Direct numerical solver baseline

DOP853 is the reference. It should always be included in the comparison figures and metrics.

The reference is expected to be more accurate than the learned policy on the cases it integrated. That is not an embarrassing result: it is the reason we have a trustworthy numerical reference.

## 41. Supervised trajectory MLP

A supervised model could learn:

\[
(\phi,r_0)\longrightarrow(u,v).
\]

It is likely to be simpler and more sample-efficient for this deterministic mapping. It is a useful comparison because it can reveal whether sequential RL was worth the extra complexity.

Do not build it until the RL implementation and evaluation are understood.

## 42. PINN

A physics-informed model could approximate u(phi,r0) and minimize the ODE residual:

\[
\mathcal R_\theta=
\frac{\partial^2u_\theta}{\partial\phi^2}
+u_\theta-\frac32u_\theta^2.
\]

It must also enforce the initial conditions. A low residual at sampled collocation points does not automatically guarantee correct global trajectories or escape behavior.

This is a later branch, not a prerequisite for your RL work.

## 43. Weak-field comparison

For large impact parameter, the leading weak-field approximation in our normalized units is:

\[
\alpha\approx\frac{2}{\bar b}.
\]

The existing src/weak_field.py provides a numerical comparison. This is a physics sanity check, not a learned model.

## 44. Near-critical rays

As r0 approaches the photon sphere from above:

\[
r_0=1.5+\delta,\qquad \delta>0,
\]

the scattering behavior becomes increasingly sensitive and the deflection can grow dramatically.

Some cases may not reach the escape event before max_angle. Such cases must be marked unresolved, not casually labeled captured.

Start near-critical experiments only after the ordinary scattering dataset, environment, and evaluation are stable.

---

# Part XII — Engineering and reproducibility

## 45. Keep experiments reproducible

For every dataset, record:
- Git commit;
- schema version;
- solver and tolerances;
- max_angle and max_step;
- delta_phi;
- closest-approach values;
- escaped and unresolved cases;
- random seed;
- split manifest.

For every training run, record:
- Git commit and dataset manifest;
- split manifest;
- observation and action definitions;
- action bounds and normalization values;
- reward formula and weights;
- algorithm and hyperparameters;
- random seed and total timesteps;
- checkpoint;
- validation and test metrics.

A model file alone is not a reproducible experiment.

## 46. Suggested future results layout

When you reach learning, keep generated artifacts out of source modules:

~~~text
outputs/
    datasets/
    figures/
    logs/
    checkpoints/
    evaluations/
~~~

Do not commit huge transient logs or checkpoints without a reason. Save configuration and summaries even when large files are kept elsewhere.

## 47. Code quality rules

- Functions should have one clear responsibility.
- Keep imports at module level.
- Do not launch simulations or training merely by importing a module.
- Use a main guard for executable scripts.
- Validate file paths and array shapes at boundaries.
- Keep units and array ordering in docstrings.
- Use explicit names such as escape_angle_radian rather than ambiguous angle.
- Never silently turn invalid numerical states into valid-looking ones.
- Explain why a non-obvious constant or tolerance exists.

---

# Part XIII — Your actual roadmap from this point

Follow this order. Do not jump directly to PPO.

| Milestone | Your deliverable | Evidence required |
|---|---|---|
| 0 | Explain the physics and data format | You can answer the Part II questions |
| 1 | Generate a fresh 30-ray dataset | NPZ + JSON manifest |
| 2 | Inspect three rays manually | Printed values and gallery |
| 3 | Create and understand ray-level splits | Split manifest and no-overlap check |
| 4 | Write the RL problem definition | Observation/action/transition/episode document |
| 5 | Implement environment reset and data selection | Manual reset output |
| 6 | Implement fixed-angle stepping | Step trace showing phi increments |
| 7 | Implement action decoding and validation | Hand-crafted action tests |
| 8 | Implement reward components individually | Component logs for hand-crafted actions |
| 9 | Finish environment termination logic | Correct terminated/truncated examples |
| 10 | Run manual diagnostic policies | Reference-copy, hold-state, random behavior plots |
| 11 | Add PPO and run a tiny smoke experiment | Saved model, load test, rollout logs |
| 12 | Inspect generated trajectories | Generated-vs-reference plots |
| 13 | Improve one design choice at a time | Ablation table and validation curves |
| 14 | Final held-out and multi-seed evaluation | CSV metrics, plots, report |
| 15 | Optional MLP/PINN/near-critical comparisons | Fair comparison using same splits |

---

# Part XIV — How to ask for help without losing the learning

When you bring code, include:

1. The file and function you are implementing.
2. What you believe the function should do.
3. What you tried.
4. The actual output or traceback.
5. What you think is wrong.

Useful requests:

- "Explain the concept first; no code yet."
- "Give me hint 1 only."
- "Review this function but don't rewrite it."
- "Point to the bug and explain why; let me fix it."
- "Give me a minimal skeleton with TODOs."
- "Now show the full solution and explain every line."
- "Quiz me on this part before we continue."

When debugging, we will distinguish:
- a Python/API mistake;
- an array-shape or indexing mistake;
- a numerical-integration issue;
- a physics-modeling mistake;
- an RL formulation issue;
- a reward-design issue;
- an evaluation/leakage issue.

Those are different classes of problems and should not be "fixed" by randomly changing code.

---

# Current session handoff

You have already reported a successful local test run:

~~~text
41 passed
~~~

The repository's physics and dataset preparation are in place. The next task is not to write PPO.

**Your next task is Milestone 1: generate a fresh 30-ray dataset under new filenames, inspect one ray manually, and show me the output.** I will then help you interpret it and give you the next small task.

The objective is that you understand every part you implement. We are building a research project and your ability to explain the decisions is part of the result.
