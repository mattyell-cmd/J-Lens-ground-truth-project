# Project
Testing J-lens decode fidelity for constructed directions on Qwen3.5-4B.

# Environment
- Work in /workspace (persists). /root is wiped on pod stop.
- export HF_HOME=/workspace/hf-cache before anything that downloads.
- Lens files: /workspace/lenses/qwen3.5-4b/{j-lens,r-lens}/lens.pt

# Rules
- I am four weeks into Python. Explain what code does, briefly, when you write it.
- Never report a result without showing the code that produced it.
- If something fails, tell me — do not silently work around it.
- Inspect data structures before assuming their shape. Print, don't guess.
- Save plots to disk as PNG, in results/.
- Research code: correct beats fast. Don't optimise unasked.
