param(
    [int]$RlSeconds = 600,
    [int]$MlpEpochs = 300,
    [int]$Seed = 42,
    [string]$RunName = "class_presentation"
)
$ErrorActionPreference = "Stop"

python -m pip install -e ".[dev,rl,ml]"
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

python -m presentation.run_parallel `
    --run-name $RunName `
    --rl-seconds $RlSeconds `
    --mlp-epochs $MlpEpochs `
    --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Parallel training or evaluation failed. Check outputs/presentation/$RunName/training_logs." }