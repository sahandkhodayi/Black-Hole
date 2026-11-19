param(
    [int]$CaseId = -1
)
$ErrorActionPreference = "Stop"

$mlp = "outputs/mlp_runs/dynamics_architecture_sweep_relu_w128_d3/best_model.pt"
$ppo = "outputs/presentation/class_presentation/selected_ppo_model.zip"
if (-not (Test-Path $mlp)) { throw "MLP checkpoint missing. Run scripts/train_for_presentation.ps1 first." }
if (-not (Test-Path $ppo)) {
    $ppo = "outputs/rl_runs/class_presentation_ppo/final_model.zip"
    if (-not (Test-Path $ppo)) { throw "PPO checkpoint missing. Run scripts/train_for_presentation.ps1 first." }
}

$argsList = @("-m", "presentation.classroom_demo", "--mlp", $mlp, "--ppo", $ppo, "--show")
if ($CaseId -ge 0) { $argsList += @("--case-id", "$CaseId") }
python @argsList
if ($LASTEXITCODE -ne 0) { throw "Classroom demo failed." }