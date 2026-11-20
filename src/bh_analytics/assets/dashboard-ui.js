function friendlyType(item) {
  if (item.kind === "presentation") return "MLP vs PPO comparison";
  if (item.kind === "mlp") return "Neural network training";
  if (item.kind === "rl") return "PPO training and test";
  return "Experiment";
}

function renderList() {
  const list = $("experiment-list");
  if (!experiments.length) {
    list.innerHTML = '<p class="muted small">No saved runs found yet.</p>';
    return;
  }
  list.innerHTML = experiments.map((item) => `
    <button class="experiment-choice ${selected?.id === item.id ? "selected" : ""}" data-id="${esc(item.id)}">
      <span class="choice-title">${esc(item.name)}</span>
      <span class="choice-type">${esc(friendlyType(item))}</span>
    </button>`).join("");
  list.querySelectorAll("[data-id]").forEach((button) => {
    button.addEventListener("click", () => selectExperiment(button.dataset.id));
  });
}

function showNoRuns() {
  $("report").disabled = true;
  $("content").innerHTML = `
    <section class="empty-state">
      <div class="empty-icon">◎</div>
      <h2>No saved experiment results found</h2>
      <p>This page does not train a model. It reads results that your training or evaluation commands have already saved.</p>
      <div class="next-steps">
        <strong>Where it looks</strong>
        <code>outputs/mlp_runs/</code>
        <code>outputs/rl_runs/</code>
        <code>outputs/presentation/</code>
      </div>
      <p class="muted">Start the dashboard from the Black Hole Lab project folder and make sure <code>--root</code> points to its <code>outputs</code> folder. No manual CSV preparation is needed.</p>
    </section>`;
}

async function refresh() {
  $("experiment-list").innerHTML = '<p class="muted">Looking for saved experiments…</p>';
  try {
    const data = await api("/api/experiments");
    experiments = data.experiments || [];
    renderList();
    if (!experiments.length) {
      selected = null;
      showNoRuns();
      return;
    }
    // The backend puts the paired comparison first; it is the easiest result to understand.
    const next = selected && experiments.find((item) => item.id === selected.id)
      ? selected.id : experiments[0].id;
    await selectExperiment(next);
  } catch (error) {
    $("experiment-list").innerHTML = "";
    $("content").innerHTML = `<section class="empty-state"><h2>Could not load results</h2><p>${esc(error.message)}</p><p class="muted">Check that the dashboard was started from the project folder with the correct outputs directory.</p></section>`;
  }
}

async function selectExperiment(id) {
  try {
    selected = await api("/api/experiment?id=" + encodeURIComponent(id));
    $("report").disabled = false;
    renderList();
    renderExperiment(selected);
  } catch (error) {
    $("content").innerHTML = `<section class="empty-state"><h2>Could not open this experiment</h2><p>${esc(error.message)}</p></section>`;
  }
}

function metricCard(label, value, explanation = "") {
  return `<div class="metric-card">
    <span class="metric-label">${esc(label)}</span>
    <strong class="metric-value">${esc(fmt(value))}</strong>
    ${explanation ? `<span class="metric-help">${esc(explanation)}</span>` : ""}
  </div>`;
}

function imageGallery(artifacts) {
  const images = (artifacts || []).filter((item) => item.kind === "image");
  if (!images.length) return "";
  return `<section class="section">
    <div class="section-heading"><div><p class="eyebrow">VISUAL RESULTS</p><h2>What the paths look like</h2></div></div>
    <div class="image-grid">${images.map((image) => `
      <a class="image-card" href="/artifact?path=${encodeURIComponent(image.path)}" target="_blank" rel="noopener">
        <img loading="lazy" src="/artifact?path=${encodeURIComponent(image.path)}" alt="${esc(image.name)}">
        <span>${esc(image.name.replace(/[_-]/g, " "))} ↗</span>
      </a>`).join("")}</div>
  </section>`;
}

function perRayTable(rows, paired = false) {
  if (!rows?.length) return '<p class="muted">Individual path scores were not saved. The overall result above is still available.</p>';
  const data = paired ? rows.map((row) => ({
    ray: row.case_id,
    model: row.model,
    error: row.normalized_rmse,
    radial: row.rmse_u,
    derivative: row.rmse_v
  })) : rows.map((row) => ({
    ray: row.case_id,
    model: "PPO",
    error: null,
    radial: row.rmse_u,
    derivative: row.rmse_v
  }));
  return `<div class="table-wrap"><table>
    <thead><tr><th>Test path</th><th>Model</th>${paired ? "<th>Overall error</th>" : ""}<th>Distance error</th><th>Change error</th></tr></thead>
    <tbody>${data.slice(0, 100).map((row) => `<tr>
      <td>${esc(row.ray)}</td><td>${esc(row.model)}</td>
      ${paired ? `<td>${esc(fmt(row.error))}</td>` : ""}
      <td>${esc(fmt(row.radial))}</td><td>${esc(fmt(row.derivative))}</td>
    </tr>`).join("")}</tbody>
  </table></div>
  <p class="table-note">Smaller errors mean the prediction stayed closer to the numerical reference.</p>`;
}

function resultBar(name, value, max, tone, display) {
  if (!isNumber(value)) return "";
  const width = max > 0 ? Math.max(1, (value / max) * 100) : 0;
  return `<div class="bar-row">
    <div class="bar-top"><span>${esc(name)}</span><strong>${esc(display ?? fmt(value))}</strong></div>
    <div class="bar-track"><div class="bar-fill ${tone}" style="width:${width}%"></div></div>
  </div>`;
}

function renderComparison(data) {
  const summary = data.summary || {};
  const mlp = data.comparison?.mlp || summary.mlp || {};
  const ppo = data.comparison?.ppo || summary.ppo || {};
  const mlpError = mlp.mean_normalized_rmse;
  const ppoError = ppo.mean_normalized_rmse;
  const rays = summary.num_test_rays ?? summary.num_rays;
  let conclusion = "The saved results include a comparison of both models.";
  let winner = null;
  let ratioText = "";
  if (isNumber(mlpError) && isNumber(ppoError)) {
    winner = mlpError <= ppoError ? "MLP" : "PPO";
    const ratio = Math.max(mlpError, ppoError) / Math.max(Math.min(mlpError, ppoError), 1e-12);
    conclusion = winner === "MLP"
      ? `The MLP followed the reference paths more closely in this test. Its average error was about ${fmt(ratio, 2)} times smaller than PPO's.`
      : `PPO followed the reference paths more closely in this test. Its average error was about ${fmt(ratio, 2)} times smaller than the MLP's.`;
    ratioText = `The lower bar is the more accurate result for these test paths.`;
  } else if (isNumber(mlp.mean_rmse_u) && isNumber(ppo.mean_rmse_u)) {
    winner = mlp.mean_rmse_u <= ppo.mean_rmse_u ? "MLP" : "PPO";
    conclusion = `${winner} had the smaller average distance-prediction error in the saved results.`;
  }
  const maxError = Math.max(isNumber(mlpError) ? mlpError : 0, isNumber(ppoError) ? ppoError : 0);
  const cards = [
    metricCard("Unseen test paths", rays, "Paths not used to train these models"),
    metricCard("MLP average error", mlpError, "Lower is better"),
    metricCard("PPO average error", ppoError, "Lower is better")
  ].join("");
  const bars = isNumber(mlpError) && isNumber(ppoError)
    ? resultBar("MLP", mlpError, maxError, "blue") + resultBar("PPO", ppoError, maxError, "violet")
    : '<p class="muted">A single overall comparison score was not saved. The per-path distance and change errors are shown below when available.</p>';
  const badge = winner ? `<span class="result-badge">${winner} performed better on this test</span>` : "";
  const testIds = (summary.test_case_ids || []).join(", ");
  $("content").innerHTML = `
    <section class="result-intro">
      <div><p class="eyebrow">MODEL COMPARISON</p><h2>${esc(data.name.replace(/[_-]/g, " "))}</h2>
        <p class="lead">${isNumber(rays) ? `Two models were tested on ${rays} complete photon paths that were kept out of training.` : "Two models were evaluated using the saved test results."}</p>
      </div>${badge}
    </section>
    <section class="metric-grid">${cards}</section>
    <section class="section result-panel">
      <p class="eyebrow">THE MAIN RESULT</p>
      <h2>${esc(winner ? winner + " was more accurate on this test" : "How the models compared")}</h2>
      <p class="conclusion">${esc(conclusion)}</p>
      <div class="bar-chart">${bars}</div>
      <p class="table-note">Prediction error measures how far each model's predicted path was from the DOP853 numerical reference. ${esc(ratioText)}</p>
    </section>
    <section class="section explainer">
      <div><p class="eyebrow">HOW TO READ THIS</p><h2>What does “error” mean?</h2></div>
      <p>For each point on a light path, the model predicts the photon's next state. We compare that prediction with the numerical reference. An error of zero would mean a perfect match; smaller numbers are better.</p>
      <p><strong>Important:</strong> this is a small test of these particular paths. It does not prove that one type of AI is always better, or that either model discovered Einstein's equations.</p>
      ${testIds ? `<p class="muted">Test path IDs: ${esc(testIds)}</p>` : ""}
    </section>
    <details class="details-panel"><summary>Show the result for each individual path</summary>${perRayTable(data.metrics, true)}</details>
    ${imageGallery(data.artifacts)}
    <p class="source-note">These results were read from the saved experiment. Opening this page does not start training or change your files.</p>`;
}

function drawTrainingChart(rows, xKey, series) {
  const width = 760, height = 250, left = 62, right = 18, top = 22, bottom = 36;
  const all = series.flatMap((s) => rows.map((row) => [row[xKey], row[s.key]])
    .filter(([x, y]) => isNumber(x) && isNumber(y)));
  if (!all.length) return '<div class="chart-empty">Training history was not saved for this run. The summary above is still available.</div>';
  const xs = all.map((point) => point[0]), ys = all.map((point) => point[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const x = (value) => left + (value - minX) / (maxX - minX || 1) * (width - left - right);
  const y = (value) => top + (maxY - value) / (maxY - minY || 1) * (height - top - bottom);
  const colors = ["#65c9db", "#a99bff", "#e8ae63"];
  const lines = series.map((s, index) => {
    const points = rows.map((row) => [row[xKey], row[s.key]])
      .filter(([xValue, yValue]) => isNumber(xValue) && isNumber(yValue));
    if (!points.length) return "";
    const coords = points.map(([px, py]) => `${x(px).toFixed(1)},${y(py).toFixed(1)}`).join(" ");
    const dots = points.length <= 80
      ? points.map(([px, py]) => `<circle cx="${x(px).toFixed(1)}" cy="${y(py).toFixed(1)}" r="3" fill="${colors[index % colors.length]}"/>`).join("")
      : "";
    return `<polyline points="${coords}" fill="none" stroke="${colors[index % colors.length]}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>${dots}`;
  }).join("");
  const grid = [0, 1, 2, 3, 4].map((i) => {
    const gy = top + i * (height - top - bottom) / 4;
    return `<line x1="${left}" y1="${gy}" x2="${width-right}" y2="${gy}" stroke="#29364b"/>`;
  }).join("");
  const legend = series.map((s, index) => `<span><i style="background:${colors[index % colors.length]}"></i>${esc(s.label)}</span>`).join("");
  return `<div class="chart-legend">${legend}</div><svg class="training-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Training progress chart">${grid}${lines}<text x="${width/2}" y="${height-7}" text-anchor="middle" fill="#a4b2c5" font-size="12">${esc(xKey === "epoch" ? "Training round" : "Training steps")}</text></svg>`;
}

function renderMlp(data) {
  const s = data.summary || {};
  const cards = [
    metricCard("Training rounds completed", s.epochs_completed, "How many passes through the training data"),
    metricCard("Best round", s.best_epoch, "The round with the best validation result"),
    metricCard("Best validation error", s.best_validation_rollout_normalized_rmse, "Lower is better"),
    metricCard("Training time", isNumber(s.elapsed_seconds) ? fmt(s.elapsed_seconds, 3) + " seconds" : null, "Time recorded by the training run")
  ].join("");
  const history = data.history || [];
  const chart = drawTrainingChart(history, "epoch", [
    { key: "train_mse", label: "Training error" },
    { key: "validation_mse", label: "Validation error" }
  ]);
  const hasPair = experiments.some((item) => item.kind === "presentation");
  $("content").innerHTML = `
    <section class="result-intro"><div><p class="eyebrow">NEURAL NETWORK TRAINING</p><h2>${esc(data.name.replace(/[_-]/g, " "))}</h2>
      <p class="lead">This run taught a neural network to predict the next point along a photon path.</p></div></section>
    <section class="metric-grid">${cards}</section>
    <section class="section result-panel"><p class="eyebrow">TRAINING PROGRESS</p><h2>Did the model learn from its examples?</h2>
      ${chart}
      <p class="table-note">Training error measures performance on examples the model learns from. Validation error uses separate examples to check whether learning carries over. Lower is generally better.</p>
    </section>
    <section class="section explainer"><p class="eyebrow">WHAT THIS RESULT TELLS YOU</p>
      <h2>${isNumber(s.best_validation_rollout_normalized_rmse) ? "The model reached a validation error of " + esc(fmt(s.best_validation_rollout_normalized_rmse)) : "Training summary"}</h2>
      <p>This page shows how training went. It does not by itself tell us how well the model predicts completely unseen full paths. That needs a separate test evaluation.</p>
      ${hasPair ? '<p class="muted">A saved MLP-vs-PPO test comparison is available in the experiment list on the left.</p>' : ""}
    </section>
    ${imageGallery(data.artifacts)}
    <p class="source-note">No CSV needs to be opened manually. The dashboard reads the training summary and history saved by the project.</p>`;
}

function renderRl(data) {
  const summary = data.summary || {};
  const cards = [
    metricCard("Test paths", summary.num_rays, "Complete photon paths evaluated"),
    metricCard("Average distance error", summary.mean_rmse_u, "Lower is better"),
    metricCard("Average change error", summary.mean_rmse_v, "Lower is better"),
    metricCard("Training steps", data.config?.timesteps_actual, "Steps taken while learning")
  ].join("");
  const history = data.history || [];
  const rewardChart = history.some((row) => isNumber(row.reward_mean) || isNumber(row.episode_reward))
    ? `<div class="training-chart-card"><h3>Reward over training</h3>${drawTrainingChart(history, "timesteps", [
        { key: "reward_mean", label: "Average reward" },
        { key: "episode_reward", label: "Latest episode" }
      ])}</div>` : "";
  const lossChart = history.some((row) => isNumber(row.policy_loss) || isNumber(row.value_loss))
    ? `<div class="training-chart-card"><h3>Model optimization</h3>${drawTrainingChart(history, "timesteps", [
        { key: "policy_loss", label: "Policy loss" },
        { key: "value_loss", label: "Value loss" }
      ])}</div>` : "";
  const validationChart = history.some((row) => isNumber(row.validation_rmse_u) || isNumber(row.validation_rmse_v))
    ? `<div class="training-chart-card"><h3>Validation prediction error</h3>${drawTrainingChart(history, "timesteps", [
        { key: "validation_rmse_u", label: "Distance (u)" },
        { key: "validation_rmse_v", label: "Change (v)" }
      ])}</div>` : "";
  const charts = rewardChart + lossChart + validationChart;
  $("content").innerHTML = `
    <section class="result-intro"><div><p class="eyebrow">PPO · REINFORCEMENT LEARNING</p><h2>${esc(data.name.replace(/[_-]/g, " "))}</h2>
      <p class="lead">This agent learned by trying to predict the next point on a photon path and receiving feedback on its predictions.</p></div></section>
    <section class="metric-grid">${cards}</section>
    <section class="section result-panel"><p class="eyebrow">LEARNING PROGRESS</p><h2>How training changed</h2>
      <div class="training-chart-grid">${charts || '<div class="chart-empty">No training curve was saved for this run.</div>'}</div>
      <p class="table-note">Reward is feedback from the training environment. Validation errors compare complete validation paths against DOP853; lower errors are better. These curves are training diagnostics, not final held-out test scores.</p>
    </section>
    <details class="details-panel"><summary>Show the result for each test path</summary>${perRayTable(data.metrics)}</details>
    ${imageGallery(data.artifacts)}
    <section class="section explainer"><h2>How good is this result?</h2>
      <p>Distance error and change error compare the PPO prediction with the numerical reference. Smaller errors are better. This is a result on the saved test paths, not a guarantee for every possible photon path.</p>
    </section>
    <p class="source-note">The dashboard reads the saved test summary directly. Individual CSV files are optional for viewing the overall result.</p>`;
}

function renderExperiment(data) {
  if (data.kind === "presentation") renderComparison(data);
  else if (data.kind === "mlp") renderMlp(data);
  else if (data.kind === "rl") renderRl(data);
  else $("content").innerHTML = '<section class="empty-state"><h2>This experiment type is not supported yet.</h2></section>';
}

$("refresh").addEventListener("click", refresh);
$("report").addEventListener("click", async () => {
  if (!selected) return;
  const button = $("report");
  button.disabled = true;
  button.textContent = "Creating report…";
  try {
    const result = await api("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: selected.id })
    });
    window.open(result.url, "_blank", "noopener");
    button.textContent = "Report is ready";
  } catch (error) {
    alert("Could not create report: " + error.message);
    button.textContent = "Save a shareable report ↗";
  } finally {
    button.disabled = false;
  }
});


function trainingIsActive(status) {
  return ["starting", "running", "stopping"].includes(status);
}

function scheduleTrainingPoll() {
  if (trainingPollTimer) clearTimeout(trainingPollTimer);
  trainingPollTimer = setTimeout(pollTrainingStatus, TRAINING_POLL_MS);
}

function renderTrainingSnapshot(snapshot) {
  const status = snapshot?.status || "idle";
  const active = trainingIsActive(status);
  const pill = $("training-status-pill");
  const startButton = $("start-training");
  const stopButton = $("stop-training");
  const message = $("training-status-message");
  pill.textContent = status.toUpperCase();
  pill.className = "training-status-pill" + (status === "running" || status === "completed"
    ? " status-" + status : status === "stopping" ? " status-stopping"
    : status === "failed" ? " status-failed" : "");
  startButton.disabled = active;
  stopButton.hidden = !active;
  stopButton.disabled = status === "stopping";

  if (status === "idle" || !snapshot?.run_name) {
    message.textContent = "No training run is active. Your saved results are below.";
    $("training-live").innerHTML = "";
    return;
  }
  const names = {
    starting: "Starting the training process",
    running: "Training is running. New graph snapshots arrive once every 60 seconds.",
    stopping: "Stop requested. The trainer will finish safely and save its current model.",
    completed: "Training finished. The model and training history have been saved.",
    failed: "Training stopped because the backend process reported an error."
  };
  message.textContent = (names[status] || "Training status") + " · " + snapshot.run_name;

  const history = snapshot.history || [];
  const latest = snapshot.latest || history[history.length - 1] || {};
  const cards = [
    metricCard("Elapsed time", isNumber(snapshot.elapsed_seconds) ? fmt(snapshot.elapsed_seconds) + " s" : null),
    metricCard("Training steps", latest.timesteps),
    metricCard("Average reward", latest.reward_mean, "Recent episode average"),
    metricCard("Validation error (u)", latest.validation_rmse_u, "Lower is better")
  ].join("");
  const chart = (title, rows, series) => {
    const present = rows.some((row) => series.some((item) => isNumber(row[item.key])));
    return present
      ? `<div class="training-chart-card"><h3>${esc(title)}</h3>${drawTrainingChart(rows, "timesteps", series)}</div>`
      : "";
  };
  const charts = [
    chart("Reward", history, [
      { key: "reward_mean", label: "Average reward" },
      { key: "episode_reward", label: "Latest episode" }
    ]),
    chart("Optimization losses", history, [
      { key: "policy_loss", label: "Policy loss" },
      { key: "value_loss", label: "Value loss" }
    ]),
    chart("Validation error", history, [
      { key: "validation_rmse_u", label: "Distance (u)" },
      { key: "validation_rmse_v", label: "Change (v)" }
    ])
  ].join("");
  let detail = "";
  if (!history.length) {
    detail = '<div class="training-empty">The trainer is starting. The first validation snapshot appears after its first PPO rollout; after that, the browser updates every minute.</div>';
  } else if (!charts) {
    detail = '<div class="training-empty">Training is progressing, but there are not enough completed episodes or validation samples to draw the graphs yet.</div>';
  } else {
    detail = `<div class="training-chart-grid">${charts}</div><p class="training-updated">Last refreshed: ${esc(new Date().toLocaleTimeString())} · Next refresh in about 1 minute</p>`;
  }
  if (status === "failed" && snapshot.log_tail?.length) {
    detail += `<div class="training-error">${esc(snapshot.log_tail.join("\\n"))}</div>`;
  }
  if (status === "completed") {
    detail += '<div class="training-empty">The final checkpoint is saved. This run is now available in the experiment list on the left.</div>';
  }
  $("training-live").innerHTML = `<div class="training-live-cards">${cards}</div>${detail}`;
}

async function pollTrainingStatus() {
  if (trainingPollTimer) clearTimeout(trainingPollTimer);
  trainingPollTimer = null;
  try {
    const wasActive = trainingSnapshot?.active === true;
    const snapshot = await api("/api/training/status");
    trainingSnapshot = snapshot;
    renderTrainingSnapshot(snapshot);
    if (trainingIsActive(snapshot.status)) {
      scheduleTrainingPoll();
    } else if (wasActive && snapshot.experiment_id) {
      await refresh();
      await selectExperiment(snapshot.experiment_id);
    }
  } catch (error) {
    $("training-status-pill").textContent = "ERROR";
    $("training-status-pill").className = "training-status-pill status-failed";
    $("training-status-message").textContent = error.message;
    $("training-live").innerHTML = `<div class="training-error">${esc(error.message)}</div>`;
  }
}

$("start-training").addEventListener("click", async () => {
  const button = $("start-training");
  button.disabled = true;
  $("training-status-message").textContent = "Preparing the training process…";
  $("training-live").innerHTML = "";
  try {
    const snapshot = await api("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        minutes: Number($("training-minutes").value),
        seed: Number($("training-seed").value)
      })
    });
    trainingSnapshot = snapshot;
    renderTrainingSnapshot(snapshot);
    scheduleTrainingPoll();
  } catch (error) {
    $("training-status-pill").textContent = "ERROR";
    $("training-status-pill").className = "training-status-pill status-failed";
    $("training-status-message").textContent = error.message;
    $("training-live").innerHTML = `<div class="training-error">${esc(error.message)}</div>`;
    button.disabled = false;
  }
});

$("stop-training").addEventListener("click", async () => {
  const button = $("stop-training");
  button.disabled = true;
  $("training-status-message").textContent = "Requesting a safe stop…";
  try {
    const snapshot = await api("/api/training/stop", { method: "POST" });
    trainingSnapshot = snapshot;
    renderTrainingSnapshot(snapshot);
    if (snapshot.active) scheduleTrainingPoll();
  } catch (error) {
    $("training-status-message").textContent = error.message;
    button.disabled = false;
  }
});

pollTrainingStatus();
