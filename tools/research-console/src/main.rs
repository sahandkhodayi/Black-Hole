use anyhow::{anyhow, Context, Result};
use eframe::egui::{self, Color32, RichText};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use serde::Deserialize;
use std::{
    collections::VecDeque,
    env,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

const MAX_HISTORY: usize = 800;

#[derive(Debug, Default, Deserialize)]
struct TelemetryEvent {
    #[serde(rename = "type", default)]
    kind: String,
    status: Option<String>,
    timesteps: Option<u64>,
    elapsed_seconds: Option<f64>,
    episode_reward: Option<f64>,
    policy_loss: Option<f64>,
    value_loss: Option<f64>,
    validation_rmse_u: Option<f64>,
    validation_rmse_v: Option<f64>,
    action_hist_u: Vec<[f64; 2]>,
    action_hist_v: Vec<[f64; 2]>,
    ray: Option<RayData>,
}

#[derive(Debug, Default, Deserialize)]
struct RayData {
    case_id: Option<i64>,
    r0: Option<f64>,
    phi: Vec<f64>,
    reference_u: Vec<f64>,
    predicted_u: Vec<f64>,
    reference_v: Vec<f64>,
    predicted_v: Vec<f64>,
}

struct ResearchConsole {
    project_root: String,
    python: String,
    run_name: String,
    seconds: String,
    seed: String,
    eval_freq: String,
    child: Option<Child>,
    run_dir: Option<PathBuf>,
    event_offset: u64,
    last_poll: Instant,
    status: String,
    timesteps: u64,
    elapsed: f64,
    rewards: VecDeque<[f64; 2]>,
    policy_losses: VecDeque<[f64; 2]>,
    value_losses: VecDeque<[f64; 2]>,
    rmse_u: VecDeque<[f64; 2]>,
    rmse_v: VecDeque<[f64; 2]>,
    ray: Option<RayData>,
    action_hist_u: Vec<[f64; 2]>,
    action_hist_v: Vec<[f64; 2]>,
    error: Option<String>,
}

impl ResearchConsole {
    fn new() -> Self {
        let root = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let python = env::var("BLACKHOLE_PYTHON").unwrap_or_else(|_| {
            if cfg!(windows) { "python".into() } else { "python3".into() }
        });
        Self {
            project_root: root.display().to_string(),
            python,
            run_name: "rust_console_demo".into(),
            seconds: "120".into(),
            seed: "42".into(),
            eval_freq: "4096".into(),
            child: None,
            run_dir: None,
            event_offset: 0,
            last_poll: Instant::now(),
            status: "Idle".into(),
            timesteps: 0,
            elapsed: 0.0,
            rewards: VecDeque::new(),
            policy_losses: VecDeque::new(),
            value_losses: VecDeque::new(),
            rmse_u: VecDeque::new(),
            rmse_v: VecDeque::new(),
            ray: None,
            action_hist_u: Vec::new(),
            action_hist_v: Vec::new(),
            error: None,
        }
    }

    fn root(&self) -> PathBuf {
        PathBuf::from(self.project_root.trim())
    }

    fn clear_run_data(&mut self) {
        self.event_offset = 0;
        self.timesteps = 0;
        self.elapsed = 0.0;
        self.rewards.clear();
        self.policy_losses.clear();
        self.value_losses.clear();
        self.rmse_u.clear();
        self.rmse_v.clear();
        self.ray = None;
        self.action_hist_u.clear();
        self.action_hist_v.clear();
        self.error = None;
    }

    fn start_training(&mut self) -> Result<()> {
        if self.child.is_some() {
            return Err(anyhow!("A training process is already running."));
        }
        let root = self.root();
        if !root.join("pyproject.toml").is_file() || !root.join("src/rl/train.py").is_file() {
            return Err(anyhow!("Project root must contain pyproject.toml and src/rl/train.py."));
        }
        let seconds: u64 = self.seconds.trim().parse().context("Seconds must be a positive integer")?;
        let seed: u64 = self.seed.trim().parse().context("Seed must be an integer")?;
        let eval_freq: u64 = self.eval_freq.trim().parse().context("Evaluation frequency must be an integer")?;
        if seconds == 0 || eval_freq == 0 {
            return Err(anyhow!("Seconds and evaluation frequency must be greater than zero."));
        }
        let run_name = self.run_name.trim();
        if run_name.is_empty() || run_name.contains('/') || run_name.contains('\\') || run_name.contains("..") {
            return Err(anyhow!("Run name must be a simple folder name."));
        }

        let run_dir = root.join("outputs").join("rl_runs").join(run_name);
        fs::create_dir_all(&run_dir).context("Could not create run output directory")?;
        fs::write(run_dir.join("console_control.json"), "{\"paused\":false,\"stop\":false}\n")?;
        let log = OpenOptions::new().create(true).append(true)
            .open(run_dir.join("training.log")).context("Could not open training log")?;
        let stderr = log.try_clone()?;
        let mut command = Command::new(self.python.trim());
        command.current_dir(&root)
            .args(["-m", "rl.train", "--seconds", &seconds.to_string(),
                   "--seed", &seed.to_string(), "--run-name", run_name,
                   "--console", "--console-eval-freq", &eval_freq.to_string()])
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(stderr));
        let child = command.spawn().with_context(|| {
            format!("Could not start Python executable '{}'. Check BLACKHOLE_PYTHON and your venv.", self.python)
        })?;
        self.clear_run_data();
        self.run_dir = Some(run_dir);
        self.child = Some(child);
        self.status = "Starting Python PPO engine…".into();
        Ok(())
    }

    fn write_control(&self, paused: bool, stop: bool) -> Result<()> {
        let dir = self.run_dir.as_ref().ok_or_else(|| anyhow!("No active run directory"))?;
        let target = dir.join("console_control.json");
        let temporary = dir.join("console_control.tmp");
        fs::write(&temporary, format!("{{\"paused\":{},\"stop\":{}}}\n", paused, stop))?;
        if target.exists() {
            fs::remove_file(&target)?;
        }
        fs::rename(temporary, target)?;
        Ok(())
    }

    fn pause_or_resume(&mut self) -> Result<()> {
        let pause = self.status != "Paused";
        self.write_control(pause, false)?;
        self.status = if pause { "Paused" } else { "Running" }.into();
        Ok(())
    }

    fn stop_training(&mut self) -> Result<()> {
        self.write_control(false, true)?;
        self.status = "Stop requested — saving model…".into();
        Ok(())
    }

    fn push_point(series: &mut VecDeque<[f64; 2]>, point: [f64; 2]) {
        series.push_back(point);
        while series.len() > MAX_HISTORY {
            series.pop_front();
        }
    }

    fn poll_events(&mut self) {
        if self.last_poll.elapsed() < Duration::from_millis(120) {
            return;
        }
        self.last_poll = Instant::now();
        if let Some(dir) = &self.run_dir {
            let path = dir.join("live_events.jsonl");
            if let Ok(mut file) = File::open(path) {
                if file.seek(SeekFrom::Start(self.event_offset)).is_ok() {
                    let mut chunk = String::new();
                    if file.read_to_string(&mut chunk).is_ok() {
                        // The Python writer may be in the middle of appending a
                        // larger validation record. Consume only newline-terminated
                        // records and retry any partial tail on the next poll.
                        if let Some(last_newline) = chunk.rfind('\n') {
                            let complete = &chunk[..=last_newline];
                            self.event_offset += complete.len() as u64;
                            for line in complete.lines().filter(|line| !line.trim().is_empty()) {
                                match serde_json::from_str::<TelemetryEvent>(line) {
                                    Ok(event) => self.apply_event(event),
                                    Err(error) => self.error = Some(format!("Telemetry parse error: {error}")),
                                }
                            }
                        }
                    }
                }
            }
        }
        let child_state = self.child.as_mut().map(|child| child.try_wait());
        match child_state {
            Some(Ok(Some(exit))) => {
                self.status = if exit.success() { "Finished".into() } else {
                    format!("Python process exited with {exit}")
                };
                self.child = None;
            }
            Some(Ok(None)) | None => {}
            Some(Err(error)) => {
                self.error = Some(format!("Could not inspect Python process: {error}"));
                self.child = None;
            }
        }
    }

    fn apply_event(&mut self, event: TelemetryEvent) {
        if let Some(steps) = event.timesteps { self.timesteps = steps; }
        if let Some(elapsed) = event.elapsed_seconds { self.elapsed = elapsed; }
        let x = self.timesteps as f64;
        if let Some(value) = event.episode_reward { Self::push_point(&mut self.rewards, [x, value]); }
        if let Some(value) = event.policy_loss { Self::push_point(&mut self.policy_losses, [x, value]); }
        if let Some(value) = event.value_loss { Self::push_point(&mut self.value_losses, [x, value]); }
        if let Some(value) = event.validation_rmse_u { Self::push_point(&mut self.rmse_u, [x, value]); }
        if let Some(value) = event.validation_rmse_v { Self::push_point(&mut self.rmse_v, [x, value]); }
        if event.ray.is_some() { self.ray = event.ray; }
        if !event.action_hist_u.is_empty() { self.action_hist_u = event.action_hist_u; }
        if !event.action_hist_v.is_empty() { self.action_hist_v = event.action_hist_v; }
        if let Some(status) = event.status {
            self.status = match status.as_str() {
                "running" if self.status != "Paused" => "Running".into(),
                "finished" => "Finished".into(),
                "stopped" => "Stopped and saved".into(),
                other => other.into(),
            };
        }
        let _ = event.kind;
    }

    fn line(ui: &mut egui::Ui, title: &str, series: &VecDeque<[f64; 2]>, height: f32) {
        ui.label(RichText::new(title).strong());
        let points = PlotPoints::from_iter(series.iter().copied());
        Plot::new(title).height(height).allow_zoom(true).allow_scroll(true)
            .show(ui, |plot_ui| {
                if !series.is_empty() {
                    plot_ui.line(Line::new(points).name(title));
                }
            });
    }

    fn orbit_plot(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Photon trajectory — normalized Schwarzschild coordinates").strong());
        Plot::new("photon_orbit").height(390.0).data_aspect(1.0)
            .legend(Legend::default()).show(ui, |plot_ui| {
                let circle = |radius: f64| -> PlotPoints {
                    PlotPoints::from_iter((0..=180).map(|i| {
                        let angle = std::f64::consts::TAU * i as f64 / 180.0;
                        [radius * angle.cos(), radius * angle.sin()]
                    }))
                };
                plot_ui.line(Line::new(circle(1.0)).name("Event horizon (1 r_s)")
                    .color(Color32::from_rgb(245, 158, 66)));
                plot_ui.line(Line::new(circle(1.5)).name("Photon sphere (1.5 r_s)")
                    .color(Color32::from_rgb(232, 183, 90)));
                if let Some(ray) = &self.ray {
                    let ref_points = ray.phi.iter().zip(ray.reference_u.iter())
                        .filter_map(|(phi, u)| if *u > 1e-5 {
                            let r = (1.0 / u).clamp(0.0, 15.0);
                            Some([r * phi.cos(), r * phi.sin()])
                        } else { None });
                    let pred_points = ray.phi.iter().zip(ray.predicted_u.iter())
                        .filter_map(|(phi, u)| if *u > 1e-5 {
                            let r = (1.0 / u).clamp(0.0, 15.0);
                            Some([r * phi.cos(), r * phi.sin()])
                        } else { None });
                    plot_ui.line(Line::new(PlotPoints::from_iter(ref_points)).name("DOP853 reference")
                        .color(Color32::from_rgb(65, 190, 255)).width(2.2));
                    plot_ui.line(Line::new(PlotPoints::from_iter(pred_points)).name("PPO policy")
                        .color(Color32::from_rgb(255, 105, 138)).width(2.0));
                }
            });
        if let Some(ray) = &self.ray {
            ui.label(format!("Validation case {}  |  r₀ = {:.3} r_s",
                ray.case_id.unwrap_or_default(), ray.r0.unwrap_or_default()));
        } else {
            ui.weak("Waiting for the first validation evaluation…");
        }
    }

    fn action_distribution_plot(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Recent normalized action distribution").strong());
        Plot::new("action_distribution").height(155.0)
            .legend(Legend::default()).show(ui, |plot_ui| {
                if !self.action_hist_u.is_empty() {
                    plot_ui.line(Line::new(PlotPoints::from_iter(
                        self.action_hist_u.iter().copied()
                    )).name("u action").color(Color32::from_rgb(65, 160, 235)));
                }
                if !self.action_hist_v.is_empty() {
                    plot_ui.line(Line::new(PlotPoints::from_iter(
                        self.action_hist_v.iter().copied()
                    )).name("v action").color(Color32::from_rgb(238, 145, 90)));
                }
            });
    }

    fn ray_component_plot(&self, ui: &mut egui::Ui, title: &str,
                          reference: &[f64], predicted: &[f64]) {
        ui.label(RichText::new(title).strong());
        Plot::new(title).height(155.0).legend(Legend::default())
            .show(ui, |plot_ui| {
                if let Some(ray) = &self.ray {
                    let ref_points = ray.phi.iter().zip(reference.iter())
                        .map(|(x, y)| [*x, *y]);
                    let pred_points = ray.phi.iter().zip(predicted.iter())
                        .map(|(x, y)| [*x, *y]);
                    plot_ui.line(Line::new(PlotPoints::from_iter(ref_points)).name("DOP853")
                        .color(Color32::from_rgb(65, 190, 255)));
                    plot_ui.line(Line::new(PlotPoints::from_iter(pred_points)).name("PPO")
                        .color(Color32::from_rgb(255, 105, 138)));
                }
            });
    }

    fn show_error(&mut self, ui: &mut egui::Ui) {
        if let Some(error) = &self.error {
            ui.colored_label(Color32::LIGHT_RED, error);
        }
    }
}

impl Drop for ResearchConsole {
    fn drop(&mut self) {
        if self.child.is_some() {
            // Closing the UI requests a graceful save from the Python callback.
            let _ = self.write_control(false, true);
        }
    }
}

impl eframe::App for ResearchConsole {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_events();
        ctx.request_repaint_after(Duration::from_millis(120));

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("BLACK HOLE LAB");
                ui.separator();
                ui.label(RichText::new(self.status.to_uppercase()).strong());
                ui.separator();
                ui.label(format!("{:,} steps", self.timesteps));
                ui.separator();
                ui.label(format!("{:.1}s", self.elapsed));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.child.is_some() {
                        if ui.button("Stop & save").clicked() {
                            if let Err(error) = self.stop_training() { self.error = Some(error.to_string()); }
                        }
                        let paused = self.status == "Paused";
                        if ui.button(if paused { "Resume" } else { "Pause" }).clicked() {
                            if let Err(error) = self.pause_or_resume() { self.error = Some(error.to_string()); }
                        }
                    }
                });
            });
        });

        egui::SidePanel::left("run_config").resizable(true).default_width(265.0).show(ctx, |ui| {
            ui.heading("Experiment");
            ui.label("Project root");
            ui.text_edit_singleline(&mut self.project_root);
            ui.label("Python executable");
            ui.text_edit_singleline(&mut self.python);
            ui.label("Run name");
            ui.text_edit_singleline(&mut self.run_name);
            ui.label("Time limit (seconds)");
            ui.text_edit_singleline(&mut self.seconds);
            ui.label("Seed");
            ui.text_edit_singleline(&mut self.seed);
            ui.label("Validation frequency");
            ui.text_edit_singleline(&mut self.eval_freq);
            ui.add_space(8.0);
            let can_start = self.child.is_none();
            if ui.add_enabled(can_start, egui::Button::new("Start PPO training")).clicked() {
                if let Err(error) = self.start_training() { self.error = Some(error.to_string()); }
            }
            ui.separator();
            ui.label("Backend");
            ui.weak("Python · Stable-Baselines3 · Gymnasium");
            ui.label("Frontend");
            ui.weak("Rust · egui · egui_plot");
            ui.separator();
            ui.weak("The native console monitors the existing Python training process. It does not change the physics, reward, policy, or optimizer.");
            self.show_error(ui);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.orbit_plot(ui);
                ui.separator();
                egui::Grid::new("metric_grid").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                    Self::line(ui, "Episode reward", &self.rewards, 155.0);
                    self.action_distribution_plot(ui);
                    ui.end_row();
                    Self::line(ui, "Policy-gradient loss", &self.policy_losses, 155.0);
                    Self::line(ui, "Value-function loss", &self.value_losses, 155.0);
                    ui.end_row();
                    Self::line(ui, "Validation RMSE — u", &self.rmse_u, 155.0);
                    Self::line(ui, "Validation RMSE — v", &self.rmse_v, 155.0);
                    ui.end_row();
                    if let Some(ray) = &self.ray {
                        let ref_u = ray.reference_u.clone();
                        let pred_u = ray.predicted_u.clone();
                        let ref_v = ray.reference_v.clone();
                        let pred_v = ray.predicted_v.clone();
                        self.ray_component_plot(ui, "Inverse radius u(φ)", &ref_u, &pred_u);
                        self.ray_component_plot(ui, "Radial derivative v(φ)", &ref_v, &pred_v);
                    } else {
                        ui.label("u(φ) reference / prediction");
                        ui.label("v(φ) reference / prediction");
                    }
                    ui.end_row();
                });
            });
        });
    }
}

fn main() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([1050.0, 700.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Black Hole Lab | Rust Research Console",
        options,
        Box::new(|_creation_context| Ok(Box::new(ResearchConsole::new()))),
    ).map_err(|error| anyhow!("Could not run native research console: {error}"))
}
