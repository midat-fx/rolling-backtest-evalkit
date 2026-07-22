# Security Policy

Report any security issue privately to midat.faizov@gmail.com rather than opening
a public issue.

## Threat model

evalkit is a local command-line tool. It makes **no network calls** and performs
**no code execution**:

- YAML configs are parsed with `yaml.safe_load` (never `yaml.load`), so a config
  cannot construct arbitrary Python objects.
- CSV data is read with pandas as data only — there is no formula or macro
  evaluation.
- No `eval`/`exec`, no `pickle`, no `subprocess`.

Configs and data files are whatever you pass on the command line; treat them as
trusted input. A malicious config could request a very large run and consume CPU
or memory, but cannot execute code.
