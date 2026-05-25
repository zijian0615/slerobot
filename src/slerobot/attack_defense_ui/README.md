# Attack / Defense UI

## Setup

```bash
cp src/slerobot/attack_defense_ui/config.example.json \
   src/slerobot/attack_defense_ui/config.json
# Edit config.json with your robot IP, HF policy path, and dataset templates.

python -m slerobot.attack_defense_ui
```

`config.json` and `session_state.json` are gitignored (machine-specific).
