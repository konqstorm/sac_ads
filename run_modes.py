from manual_mode import run_manual
from baseline_mode import run_baseline
from agent_mode import run_agent
from runtime_options import (
    load_config,
    resolve_mode,
    resolve_renderer,
    resolve_stochastic_agent,
)


if __name__ == "__main__":
    cfg_path = "config.yaml"
    cfg = load_config(cfg_path)
    mode = resolve_mode(cfg)
    renderer = resolve_renderer(cfg)
    stochastic_agent = resolve_stochastic_agent(cfg, stochastic=None, default=True)

    if mode == "manual":
        run_manual(cfg_path=cfg_path, renderer=renderer)
    elif mode == "baseline":
        run_baseline(cfg_path=cfg_path, renderer=renderer)
    elif mode == "agent":
        run_agent(cfg_path=cfg_path, stochastic=stochastic_agent, renderer=renderer)
    else:
        raise SystemExit("Unknown mode. Use 'manual', 'baseline', or 'agent'.")
