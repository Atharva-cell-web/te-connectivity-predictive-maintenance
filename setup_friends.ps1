# Quick Setup for your Friends

# 1. Install Node.js & Ollama
winget install OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements

# 2. Install Claude Code
npm install -g @anthropic-ai/claude-code

# 3. Setup the local "Opus" model (Qwen2.5-Coder 7B)
ollama pull qwen2.5-coder:7b
"FROM qwen2.5-coder:7b" | Out-File -FilePath Modelfile -Encoding utf8
ollama create opus -f Modelfile
ollama create claude-opus-4-6 -f Modelfile

# 4. Redirect Claude to Local Ollama
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_BASE_URL", "http://localhost:11434", "User")
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_AUTH_TOKEN", "dummy-token", "User")

# 5. Final Step: Logout to enable the trick
claude /logout

# Restart terminal and run: claude --model opus
