(() => {
  const canvas = document.getElementById("orbit");
  const ctx = canvas.getContext("2d");
  const slider = document.getElementById("closest");
  const output = document.getElementById("closest-value");
  const runButton = document.getElementById("simulate");
  const errorNode = document.getElementById("error");
  let trajectory = null;

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function draw() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2, scale = Math.min(w, h) / 54;
    ctx.strokeStyle = "rgba(125,157,190,.10)";
    ctx.lineWidth = 1;
    for (let r = 5; r <= 25; r += 5) {
      ctx.beginPath(); ctx.arc(cx, cy, r * scale, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
    ctx.setLineDash([5, 5]); ctx.strokeStyle = "#b49aff"; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(cx, cy, 1.5 * scale, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    const glow = ctx.createRadialGradient(cx, cy, 2, cx, cy, 1.8 * scale);
    glow.addColorStop(0, "#02040a"); glow.addColorStop(.65, "#050812"); glow.addColorStop(1, "#f1b85b");
    ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(cx, cy, 1.0 * scale, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#f1b85b"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(cx, cy, scale, 0, Math.PI * 2); ctx.stroke();
    if (trajectory && trajectory.x.length > 1) {
      ctx.save(); ctx.beginPath();
      trajectory.x.forEach((x, i) => {
        const px = cx + x * scale, py = cy - trajectory.y[i] * scale;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = "#65d6ef"; ctx.lineWidth = 2.4; ctx.lineJoin = "round"; ctx.shadowColor = "#65d6ef"; ctx.shadowBlur = 10; ctx.stroke(); ctx.restore();
      const first = trajectory.x.length - 1;
      ctx.fillStyle = "#e9faff"; ctx.beginPath(); ctx.arc(cx + trajectory.x[first] * scale, cy - trajectory.y[first] * scale, 3.2, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = "#8497b1"; ctx.font = "10px system-ui";
    ctx.fillText("rₛ = 1", cx + 9, cy - scale - 7);
    ctx.fillStyle = "#b49aff"; ctx.fillText("photon sphere · 1.5 rₛ", cx + 1.8 * scale, cy - 1.5 * scale);
  }

  function setRadius(value) {
    slider.value = value;
    output.textContent = Number(value).toFixed(2) + " rₛ";
    document.getElementById("radius-metric").textContent = Number(value).toFixed(2) + " rₛ";
    document.querySelectorAll("[data-radius]").forEach(button => button.classList.toggle("selected", Number(button.dataset.radius) === Number(value)));
  }

  async function simulate() {
    const closest = Number(slider.value);
    runButton.disabled = true;
    runButton.innerHTML = "<span>…</span> Integrating geodesic";
    errorNode.textContent = "";
    document.getElementById("solver-status").innerHTML = "<i></i> SOLVING · DOP853";
    try {
      const response = await fetch("/api/physics/simulate", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({closest_approach: closest})
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Simulation failed.");
      trajectory = result;
      document.getElementById("deflection").textContent = result.escaped ? result.deflection_degrees.toFixed(3) + "°" : "Not escaped";
      document.getElementById("nfev").textContent = result.nfev.toLocaleString();
      document.getElementById("outcome").textContent = result.escaped ? "Scattered" : "Integration limit";
      document.getElementById("point-count").textContent = result.x.length.toLocaleString() + " plotted samples";
      document.getElementById("solver-status").innerHTML = "<i></i> SOLUTION COMPLETE";
      draw();
    } catch (error) {
      errorNode.textContent = error.message;
      document.getElementById("solver-status").innerHTML = "<i></i> SOLVER ERROR";
    } finally {
      runButton.disabled = false;
      runButton.innerHTML = "<span>↻</span> Run simulation";
    }
  }

  slider.addEventListener("input", () => setRadius(slider.value));
  slider.addEventListener("change", simulate);
  runButton.addEventListener("click", simulate);
  document.querySelectorAll("[data-radius]").forEach(button => button.addEventListener("click", () => { setRadius(button.dataset.radius); simulate(); }));
  if ("ResizeObserver" in window) new ResizeObserver(resizeCanvas).observe(canvas);
  else window.addEventListener("resize", resizeCanvas);
  setRadius(slider.value);
  resizeCanvas();
  simulate();
})();
