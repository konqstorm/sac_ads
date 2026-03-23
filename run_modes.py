from manual_mode import run_manual
from baseline_mode import run_baseline
from agent_mode import run_agent


if __name__ == "__main__":
    mode = "manual"  # manual | baseline | agent

    if mode == "manual":
        run_manual()
    elif mode == "baseline":
        run_baseline()
    elif mode == "agent":
        run_agent()
    else:
        raise SystemExit("Unknown mode. Use 'manual', 'baseline', or 'agent'.")
