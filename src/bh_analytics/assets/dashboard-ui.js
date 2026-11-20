let physicsRay = null;
let physicsIndex = 0;
let physicsDirty = false;
let physicsProgress = 1;
let physicsAnimation = null;
let trainingVisual = null;

function canvasContext(canvas) {
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * ratio);
  const pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function orbitPoints(phi, u) {
  const points = [];
  const length = Math.min(phi?.length || 0, u?.length || 0);
  for (let index = 0; index < length; index += 1) {
    const angle = Number(phi[index]);
    const inverseRadius = Number(u[index]);
    if (!Number.isFinite(angle) || !Number.isFinite(inverseRadius)) continue;
    const radius = inverseRadius > 1e-5 ? Math.min(1 / inverseRadius, 20) : 20;
    points.push({ x: radius * Math.cos(angle), y: radius * Math.sin(angle) });
  }
  return points;
}

function drawOrbitCanvas(canvas, data, options = {}) {
  const surface = canvasContext(canvas);
  if (!surface) return;
  const { ctx, width, height } = surface;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070e19";
  ctx.fillRect(0, 0, width, height);

  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width * 0.455, height * 0.455) / 20;
  const px = (value) => centerX + value * scale;
  const py = (value) => centerY - value * scale;

  ctx.save();
  ctx.strokeStyle = "rgba(130, 158, 188, .11)";
  ctx.lineWidth = 1;
  for (const radius of [5, 10, 15, 20]) {
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius * scale, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.setLineDash([3, 6]);
  ctx.beginPath();
  ctx.moveTo(0, centerY); ctx.lineTo(width, centerY);
  ctx.moveTo(centerX, 0); ctx.lineTo(centerX, height);
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = "rgba(255, 184, 107, .76)";
  ctx.setLineDash([5, 5]);
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.arc(centerX, centerY, 1.5 * scale, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#03060b";
  ctx.beginPath();
  ctx.arc(centerX, centerY, Math.max(1, scale), 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ffb86b";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.shadowColor = "rgba(255, 144, 63, .3)";
  ctx.shadowBlur = 20;
  ctx.beginPath();
  ctx.arc(centerX, centerY, Math.max(1, scale * 1.16), 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  ctx.font = "10px Segoe UI, Arial";
  ctx.fillStyle = "rgba(255, 198, 138, .88)";
  ctx.fillText("photon sphere · 1.5 rₛ", centerX + 1.5 * scale + 7, centerY - 1.5 * scale - 5);
  ctx.fillStyle = "rgba(255, 198, 138, .75)";
  ctx.fillText("event horizon", centerX + scale + 7, centerY + 14);

  const reference = orbitPoints(data?.phi, data?.u);
  const prediction = orbitPoints(data?.phi, options.predictedU);
  const drawPath = (points, color, dashed, progress, lineWidth) => {
    if (!points.length) return null;
    const count = Math.max(2, Math.min(points.length, Math.floor((points.length - 1) * progress) + 1));
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (dashed) ctx.setLineDash([7, 5]);
    ctx.beginPath();
    ctx.moveTo(px(points[0].x), py(points[0].y));
    for (let index = 1; index < count; index += 1) {
      ctx.lineTo(px(points[index].x), py(points[index].y));
    }
    ctx.stroke();
    ctx.restore();
    return { point: points[count - 1], count };
  };

  const color = options.referenceColor || "#ffb86b";
  const progress = Number.isFinite(options.progress) ? options.progress : 1;
  const referenceResult = drawPath(reference, color, false, progress, 2.7);
  if (prediction.length) drawPath(prediction, "#f58da5", true, 1, 2.2);
  const marker = Number.isInteger(options.markerIndex) && reference.length
    ? reference[Math.max(0, Math.min(reference.length - 1, options.markerIndex))]
    : referenceResult?.point;

  if (marker) {
    ctx.save();
    ctx.fillStyle = "#f6fbff";
    ctx.shadowColor = color;
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.arc(px(marker.x), py(marker.y), 4.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
  if (prediction.length && options.showPredictionDot) {
    const point = prediction[prediction.length - 1];
    ctx.fillStyle = "#f58da5";
    ctx.beginPath();
    ctx.arc(px(point.x), py(point.y), 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
  if (!reference.length) {
    ctx.fillStyle = "#9daec4";
    ctx.font = "12px Segoe UI, Arial";
    ctx.textAlign = "center";
    ctx.fillText("Waiting for a trajectory…", centerX, centerY - 15);
  }
}

function drawStateCanvas(canvas, xValues, yValues, activeIndex, color, zeroLine = false) {
  const surface = canvasContext(canvas);
  if (!surface) return;
  const { ctx, width, height } = surface;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0b1625";
  ctx.fillRect(0, 0, width, height);

  const values = yValues.filter(Number.isFinite);
  if (values.length < 2 || xValues.length < 2) {
    ctx.fillStyle = "#8092aa";
    ctx.font = "11px Segoe UI, Arial";
    ctx.fillText("No state samples", 12, 24);
    return;
  }

  let minY = Math.min(...values);
  let maxY = Math.max(...values);
  if (zeroLine) { minY = Math.min(minY, 0); maxY = Math.max(maxY, 0); }
  const padding = Math.max((maxY - minY) * 0.14, 0.015);
  minY -= padding; maxY += padding;
  const left = 42, right = 12, top = 12, bottom = 25;
  const minX = xValues[0], maxX = xValues[xValues.length - 1];
  const x = (value) => left + (value - minX) / (maxX - minX || 1) * (width - left - right);
  const y = (value) => top + (maxY - value) / (maxY - minY || 1) * (height - top - bottom);

  ctx.font = "9px Segoe UI, Arial";
  ctx.fillStyle = "#8092aa";
  ctx.strokeStyle = "rgba(130, 158, 188, .15)";
  for (let step = 0; step <= 3; step += 1) {
    const value = maxY - (maxY - minY) * step / 3;
    const yy = y(value);
    ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(width - right, yy); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(value.toFixed(2), left - 6, yy + 3);
  }
  ctx.textAlign = "center";
  ctx.fillText("φ (radians)", (left + width - right) / 2, height - 5);

  if (zeroLine && minY < 0 && maxY > 0) {
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,.25)";
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(left, y(0)); ctx.lineTo(width - right, y(0)); ctx.stroke();
    ctx.restore();
  }

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  let started = false;
  const count = Math.min(xValues.length, yValues.length);
  for (let index = 0; index < count; index += 1) {
    if (!Number.isFinite(xValues[index]) || !Number.isFinite(yValues[index])) continue;
    if (!started) { ctx.moveTo(x(xValues[index]), y(yValues[index])); started = true; }
    else ctx.lineTo(x(xValues[index]), y(yValues[index]));
  }
  ctx.stroke();
  ctx.restore();

  if (Number.isInteger(activeIndex) && activeIndex >= 0 && activeIndex < count) {
    const cx = x(xValues[activeIndex]);
    const cy = y(yValues[activeIndex]);
    ctx.save();
    ctx.strokeStyle = "rgba(255, 255, 255, .28)";
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(cx, top); ctx.lineTo(cx, height - bottom); ctx.stroke();
    ctx.fillStyle = "#ffffff"; ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, 4.3, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.restore();
  }
}

function drawPhysicsViews() {
  if (!physicsRay) return;
  drawOrbitCanvas($("physics-orbit"), physicsRay, {
    progress: physicsProgress,
    markerIndex: physicsIndex,
  });
  drawStateCanvas($("u-plot"), physicsRay.phi, physicsRay.u, physicsIndex, "#ffb86b");
  drawStateCanvas($("v-plot"), physicsRay.phi, physicsRay.v, physicsIndex, "#65d2ee", true);
}

function updatePhysicsReadouts() {
  if (!physicsRay) return;
  $("deflection-value").textContent = isNumber(physicsRay.deflection_degrees)
    ? fmt(physicsRay.deflection_degrees, 4) + "°" : "Not escaped";
  $("samples-value").textContent = fmt(physicsRay.samples, 0);
  const radius = physicsRay.closest_approach;
  $("physics-insight").textContent = radius < 2.4
    ? "This ray passes close to the photon sphere, so its direction changes sharply before it escapes."
    : radius < 5
      ? "The photon is strongly deflected. Increasing r₀ moves the path farther from the black hole."
      : "Farther from the black hole, spacetime curvature has a weaker effect and the deflection becomes smaller.";
  $("solver-state").textContent = "Solved · " + fmt(physicsRay.deflection_degrees, 3) + "° deflection";
}

async function tracePhysicsRay() {
  const button = $("trace-ray");
  const radius = Number($("closest-approach").value);
  button.disabled = true;
  button.textContent = "Solving…";
  $("solver-state").textContent = "Integrating the ODE…";
  try {
    const ray = await api("/api/physics/ray?r0=" + encodeURIComponent(radius));
    physicsRay = ray;
    physicsIndex = Math.floor(ray.phi.length / 2);
    physicsProgress = 1;
    physicsDirty = false;
    $("r0-value").textContent = radius.toFixed(2);
    $("play-ray").disabled = false;
    $("play-ray").textContent = "▶ Animate photon";
    updatePhysicsReadouts();
    drawPhysicsViews();
  } catch (error) {
    $("solver-state").textContent = "Solver error";
    $("physics-insight").textContent = error.message;
  } finally {
    button.disabled = false;
    button.innerHTML = "Trace this path <span>↗</span>";
  }
}

function stopPhysicsAnimation() {
  if (physicsAnimation !== null) cancelAnimationFrame(physicsAnimation);
  physicsAnimation = null;
  $("play-ray").textContent = physicsProgress >= 1 ? "↻ Replay photon" : "▶ Resume photon";
}

function animatePhysics() {
  const duration = 6500;
  const startTime = performance.now() - physicsProgress * duration;
  const frame = (now) => {
    physicsProgress = Math.min(1, (now - startTime) / duration);
    physicsIndex = Math.min(physicsRay.phi.length - 1,
      Math.floor(physicsProgress * (physicsRay.phi.length - 1)));
    drawPhysicsViews();
    if (physicsProgress < 1) physicsAnimation = requestAnimationFrame(frame);
    else {
      physicsAnimation = null;
      $("play-ray").textContent = "↻ Replay photon";
    }
  };
  physicsAnimation = requestAnimationFrame(frame);
}

$("closest-approach").addEventListener("input", (event) => {
  $("r0-value").textContent = Number(event.target.value).toFixed(2);
  physicsDirty = true;
  $("solver-state").textContent = "New starting condition · trace to update";
  $("play-ray").disabled = true;
});
$("trace-ray").addEventListener("click", tracePhysicsRay);
$("play-ray").addEventListener("click", () => {
  if (!physicsRay || physicsDirty) return;
  if (physicsAnimation !== null) { stopPhysicsAnimation(); return; }
  if (physicsProgress >= 1) {
    physicsProgress = 0;
    physicsIndex = 0;
  }
  $("play-ray").textContent = "Ⅱ Pause animation";
  animatePhysics();
});

function scheduleTrainingPoll() {
  if (trainingPollTimer !== null) clearTimeout(trainingPollTimer);
  trainingPollTimer = setTimeout(pollTrainingStatus, TRAINING_POLL_MS);
}

function drawRewardChart(rows) {
  const surface = canvasContext($("reward-plot"));
  if (!surface) return;
  const { ctx, width, height } = surface;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0b1625"; ctx.fillRect(0, 0, width, height);
  const points = (rows || []).filter((row) => isNumber(row.timesteps) && isNumber(row.reward_mean));
  if (points.length < 2) {
    ctx.fillStyle = "#8092aa"; ctx.font = "11px Segoe UI, Arial";
    ctx.fillText("Reward curve appears when complete training episodes are recorded.", 10, 25);
    return;
  }
  const minX = points[0].timesteps, maxX = points[points.length - 1].timesteps;
  const rewards = points.map((row) => row.reward_mean);
  const minY = Math.min(...rewards), maxY = Math.max(...rewards);
  const left = 39, right = 12, top = 12, bottom = 23;
  const x = (value) => left + (value - minX) / (maxX - minX || 1) * (width - left - right);
  const y = (value) => top + (maxY - value) / (maxY - minY || 1) * (height - top - bottom);
  ctx.strokeStyle = "rgba(130,158,188,.16)"; ctx.fillStyle = "#8092aa";
  for (let step = 0; step <= 3; step += 1) {
    const yy = top + step * (height - top - bottom) / 3;
    ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(width - right, yy); ctx.stroke();
  }
  ctx.strokeStyle = "#65d2ee"; ctx.lineWidth = 2.3; ctx.beginPath();
  points.forEach((row, index) => {
    if (index === 0) ctx.moveTo(x(row.timesteps), y(row.reward_mean));
    else ctx.lineTo(x(row.timesteps), y(row.reward_mean));
  });
  ctx.stroke();
  ctx.font = "9px Segoe UI, Arial"; ctx.textAlign = "left";
  ctx.fillText("early training", left, height - 5);
  ctx.textAlign = "right"; ctx.fillText("recent", width - right, height - 5);
}

function trainingReadout(label, value) {
  return '<div class="training-readout"><span>' + esc(label) +
    '</span><strong>' + esc(value) + '</strong></div>';
}

function renderTrainingSnapshot(snapshot) {
  const status = snapshot?.status || "idle";
  const active = ["starting", "running", "stopping"].includes(status);
  const pill = $("training-status-pill");
  const startButton = $("start-training");
  const stopButton = $("stop-training");
  pill.textContent = status.toUpperCase();
  pill.className = "status-pill" + (["running", "completed", "stopping", "failed"].includes(status)
    ? " status-" + status : "");
  startButton.disabled = active;
  stopButton.hidden = !active;
  stopButton.disabled = status === "stopping";

  if (!snapshot?.run_name) {
    $("training-status-message").textContent = "No active training run.";
    $("training-ray-label").textContent = "Waiting for a training run";
    $("training-visual-note").textContent = "Start training to see the agent's predicted orbit overlaid on the numerical solution. The comparison updates after each validation check.";
    $("training-readouts").innerHTML = "";
    drawOrbitCanvas($("training-orbit"), null);
    drawRewardChart([]);
    return;
  }

  const messages = {
    starting: "Starting the PPO process.",
    running: "The agent is training. Visual snapshots refresh once per minute.",
    stopping: "Stop requested. The current model is being saved.",
    completed: "Training finished. The final model is saved.",
    failed: "Training failed. Check the run log below."
  };
  $("training-status-message").textContent = messages[status] || "Training status updated.";
  const latest = snapshot.latest || {};
  $("training-readouts").innerHTML = [
    trainingReadout("Training steps", isNumber(latest.timesteps) ? fmt(latest.timesteps, 0) : "Starting"),
    trainingReadout("Average reward", isNumber(latest.reward_mean) ? fmt(latest.reward_mean, 4) : "Not yet recorded"),
    trainingReadout("Validation RMSE · u", isNumber(latest.validation_rmse_u) ? fmt(latest.validation_rmse_u, 4) : "Not yet recorded"),
    trainingReadout("Validation RMSE · v", isNumber(latest.validation_rmse_v) ? fmt(latest.validation_rmse_v, 4) : "Not yet recorded")
  ].join("");

  trainingVisual = snapshot.visual || null;
  if (trainingVisual && trainingVisual.phi?.length) {
    drawOrbitCanvas($("training-orbit"), {
      phi: trainingVisual.phi, u: trainingVisual.reference_u,
    }, {
      predictedU: trainingVisual.predicted_u,
      referenceColor: "#65d2ee",
      showPredictionDot: true,
    });
    $("training-ray-label").textContent =
      "Validation ray " + trainingVisual.case_id + " · r₀ " +
      fmt(trainingVisual.closest_approach, 3) + " rₛ";
    $("training-visual-note").textContent =
      "At " + fmt(trainingVisual.timesteps, 0) + " training steps, both curves show the same ray. " +
      "Validation RMSE: u = " + fmt(trainingVisual.validation_rmse_u, 4) +
      ", v = " + fmt(trainingVisual.validation_rmse_v, 4) +
      ". The dashed curve is the agent's own rollout.";
  } else {
    drawOrbitCanvas($("training-orbit"), null);
    $("training-ray-label").textContent = "Waiting for first validation snapshot";
    $("training-visual-note").textContent =
      "The policy comparison appears after the trainer completes its first validation rollout. Later snapshots update about once per minute.";
  }

  drawRewardChart(snapshot.history || []);
  if (status === "failed" && snapshot.log_tail?.length) {
    $("training-visual-note").textContent = snapshot.log_tail.join("\n");
    $("training-visual-note").classList.add("error-message");
  } else {
    $("training-visual-note").classList.remove("error-message");
  }
  if (status === "completed") {
    $("training-ray-label").textContent = "Final validation snapshot · run complete";
    if (snapshot.experiment_id) refreshExperiments(snapshot.experiment_id);
  }
}

async function pollTrainingStatus() {
  if (trainingPollTimer !== null) clearTimeout(trainingPollTimer);
  trainingPollTimer = null;
  try {
    const previousActive = trainingSnapshot?.active === true;
    const snapshot = await api("/api/training/status");
    trainingSnapshot = snapshot;
    renderTrainingSnapshot(snapshot);
    if (["starting", "running", "stopping"].includes(snapshot.status)) scheduleTrainingPoll();
    else if (previousActive && snapshot.experiment_id) await refreshExperiments(snapshot.experiment_id);
  } catch (error) {
    $("training-status-pill").textContent = "API ERROR";
    $("training-status-pill").className = "status-pill status-failed";
    $("training-status-message").textContent = error.message;
  }
}

$("start-training").addEventListener("click", async () => {
  const button = $("start-training");
  button.disabled = true;
  $("training-status-pill").textContent = "STARTING";
  $("training-status-message").textContent = "Checking the dataset and starting PPO…";
  $("training-visual-note").classList.remove("error-message");
  try {
    const snapshot = await api("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        minutes: Number($("training-minutes").value),
        seed: Number($("training-seed").value),
      }),
    });
    trainingSnapshot = snapshot;
    renderTrainingSnapshot(snapshot);
    scheduleTrainingPoll();
  } catch (error) {
    $("training-status-pill").textContent = "ERROR";
    $("training-status-pill").className = "status-pill status-failed";
    $("training-status-message").textContent = error.message;
    button.disabled = false;
  }
});

$("stop-training").addEventListener("click", async () => {
  const button = $("stop-training");
  button.disabled = true;
  $("training-status-message").textContent = "Sending a safe stop request…";
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

function metricCard(label, value, help = "") {
  return '<div class="metric-card"><span class="metric-label">' + esc(label) +
    '</span><strong class="metric-value">' + esc(fmt(value)) +
    '</strong><span class="metric-help">' + esc(help) + '</span></div>';
}

function renderSavedExperiment(data) {
  const summary = data.summary || {};
  const cards = [];
  let title = "Saved experiment";
  let explanation = "A result saved by a previous training or evaluation run.";
  if (data.kind === "presentation") {
    title = "MLP vs PPO · held-out test";
    explanation = "Both models were compared on the same photon paths kept out of training.";
    const mlp = data.comparison?.mlp || summary.mlp || {};
    const ppo = data.comparison?.ppo || summary.ppo || {};
    cards.push(metricCard("Unseen test paths", summary.num_test_rays ?? summary.num_rays));
    cards.push(metricCard("MLP average error", mlp.mean_normalized_rmse, "Lower is better"));
    cards.push(metricCard("PPO average error", ppo.mean_normalized_rmse, "Lower is better"));
  } else if (data.kind === "mlp") {
    title = "Supervised MLP · training run";
    explanation = "The network learns to predict the next inverse-radius state.";
    cards.push(metricCard("Training rounds", summary.epochs_completed));
    cards.push(metricCard("Best round", summary.best_epoch));
    cards.push(metricCard("Best validation error", summary.best_validation_rollout_normalized_rmse));
  } else if (data.kind === "rl") {
    title = "PPO · saved training run";
    explanation = "The agent was trained to predict the next point on a reference photon trajectory.";
    cards.push(metricCard("Test paths", summary.num_rays));
    cards.push(metricCard("Distance error", summary.mean_rmse_u));
    cards.push(metricCard("Change error", summary.mean_rmse_v));
  }
  const images = (data.artifacts || []).filter((item) => item.kind === "image");
  const gallery = images.length ? '<div class="image-grid">' + images.map((item) =>
    '<a class="image-card" href="/artifact?path=' + encodeURIComponent(item.path) +
    '" target="_blank" rel="noopener"><img loading="lazy" src="/artifact?path=' +
    encodeURIComponent(item.path) + '" alt="' + esc(item.name) + '"><span>' +
    esc(item.name.replace(/[_-]/g, " ")) + '</span></a>').join("") + '</div>' : "";

  let winner = null, bars = "";
  if (data.kind === "presentation") {
    const mlp = data.comparison?.mlp || summary.mlp || {};
    const ppo = data.comparison?.ppo || summary.ppo || {};
    const mlpError = mlp.mean_normalized_rmse;
    const ppoError = ppo.mean_normalized_rmse;
    if (isNumber(mlpError) && isNumber(ppoError)) {
      winner = mlpError <= ppoError ? "MLP" : "PPO";
      const max = Math.max(mlpError, ppoError, 1e-12);
      bars =
        '<div class="bar-row"><div class="bar-top"><span>MLP</span><strong>' + fmt(mlpError, 4) +
        '</strong></div><div class="bar-track"><div class="bar-fill blue" style="width:' +
        (mlpError / max * 100) + '%"></div></div></div>' +
        '<div class="bar-row"><div class="bar-top"><span>PPO</span><strong>' + fmt(ppoError, 4) +
        '</strong></div><div class="bar-track"><div class="bar-fill violet" style="width:' +
        (ppoError / max * 100) + '%"></div></div></div>';
    }
  }

  $("content").innerHTML =
    '<div class="saved-intro"><h3>' + esc(data.name.replace(/[_-]/g, " ")) +
    ' · ' + esc(title) + '</h3><p>' + esc(explanation) + '</p></div>' +
    '<div class="metric-grid">' + cards.join("") + '</div>' + bars +
    (gallery || '<p class="muted">No saved plot images were found for this run.</p>') +
    (winner ? '<p class="table-note">' + winner +
      ' had the smaller average normalized error on this saved test. This describes these test paths only.</p>' : "");
}

async function refreshExperiments(selectId = null) {
  try {
    const data = await api("/api/experiments");
    experiments = data.experiments || [];
    const select = $("experiment-select");
    const preferred = selectId || selected?.id || experiments[0]?.id || "";
    select.innerHTML = experiments.length
      ? experiments.map((item) => '<option value="' + esc(item.id) + '">' +
          esc(item.name) + ' · ' + esc(item.label) + '</option>').join("")
      : '<option value="">No saved experiments yet</option>';
    select.disabled = experiments.length === 0;
    $("report").disabled = true;
    if (!experiments.length) {
      selected = null;
      $("content").innerHTML = '<p class="empty-state">Train a model or run an evaluation. Saved results will appear here.</p>';
      return;
    }
    const chosen = experiments.some((item) => item.id === preferred) ? preferred : experiments[0].id;
    select.value = chosen;
    await selectExperiment(chosen);
  } catch (error) {
    $("content").innerHTML = '<p class="error-message">' + esc(error.message) + '</p>';
  }
}

async function selectExperiment(id) {
  if (!id) return;
  try {
    selected = await api("/api/experiment?id=" + encodeURIComponent(id));
    $("experiment-select").value = id;
    $("report").disabled = false;
    renderSavedExperiment(selected);
  } catch (error) {
    $("content").innerHTML = '<p class="error-message">' + esc(error.message) + '</p>';
  }
}

$("experiment-select").addEventListener("change", (event) => selectExperiment(event.target.value));
$("refresh").addEventListener("click", () => refreshExperiments());

$("report").addEventListener("click", async () => {
  if (!selected) return;
  const button = $("report");
  button.disabled = true;
  button.textContent = "Creating…";
  try {
    const result = await api("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: selected.id }),
    });
    window.open(result.url, "_blank", "noopener");
  } catch (error) {
    $("content").insertAdjacentHTML("afterbegin", '<p class="error-message">' + esc(error.message) + '</p>');
  } finally {
    button.disabled = false;
    button.textContent = "Save HTML report";
  }
});

window.addEventListener("resize", () => {
  if (physicsRay) drawPhysicsViews();
  if (trainingVisual) drawOrbitCanvas($("training-orbit"), {
    phi: trainingVisual.phi, u: trainingVisual.reference_u,
  }, {
    predictedU: trainingVisual.predicted_u,
    referenceColor: "#65d2ee",
    showPredictionDot: true,
  });
});

async function initializeClassroom() {
  await tracePhysicsRay();
  await refreshExperiments();
  await pollTrainingStatus();
}
initializeClassroom();
