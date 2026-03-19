# Run this script after Qwen2.5-Coder 7B finishes downloading!
Write-Host "Creating Opus and claude-opus-4-6 aliases for Qwen2.5-Coder 7B..."

# 1. Create the Modelfile
"FROM qwen2.5-coder:7b" | Out-File -FilePath Modelfile -Encoding utf8

# 2. Use the Ollama executable on D: drive to create them
$ollamaPath = "D:\Ollama\ollama.exe"
if (-Not (Test-Path $ollamaPath)) {
    $ollamaPath = "ollama"
}

& $ollamaPath create opus -f Modelfile
& $ollamaPath create claude-opus-4-6 -f Modelfile

Write-Host "Aliases created successfully!"
