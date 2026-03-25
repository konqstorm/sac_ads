import yaml


SUPPORTED_MODES = {"manual", "baseline", "agent"}
SUPPORTED_RENDERERS = {"2d", "3d"}


def load_config(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _run_cfg(cfg):
    if not isinstance(cfg, dict):
        return {}
    run_cfg = cfg.get("run", {})
    return run_cfg if isinstance(run_cfg, dict) else {}


def resolve_mode(cfg, mode=None):
    selected = mode if mode is not None else _run_cfg(cfg).get("mode", "manual")
    selected = str(selected).strip().lower()
    if selected not in SUPPORTED_MODES:
        raise ValueError(
            f"Unknown run mode '{selected}'. Supported modes: {sorted(SUPPORTED_MODES)}."
        )
    return selected


def resolve_renderer(cfg, renderer=None):
    selected = renderer if renderer is not None else _run_cfg(cfg).get("renderer", "2d")
    selected = str(selected).strip().lower()
    if selected not in SUPPORTED_RENDERERS:
        raise ValueError(
            f"Unknown renderer '{selected}'. Supported renderers: {sorted(SUPPORTED_RENDERERS)}."
        )
    return selected


def resolve_fps(cfg, default=30):
    fps = _run_cfg(cfg).get("fps", default)
    try:
        fps = int(fps)
    except (TypeError, ValueError):
        fps = default
    return max(1, fps)


def resolve_stochastic_agent(cfg, stochastic=None, default=True):
    if stochastic is None:
        value = _run_cfg(cfg).get("stochastic_agent", default)
    else:
        value = stochastic
    return bool(value)


def resolve_do_gif(cfg, default=False):
    value = _run_cfg(cfg).get("do_gif", default)
    return bool(value)


def resolve_gif_directory(cfg, default="tmp_gif"):
    value = _run_cfg(cfg).get("gif_directory", default)
    value = str(value).strip() if value is not None else ""
    return value or default


def resolve_gif_name(cfg, default="run.gif"):
    value = _run_cfg(cfg).get("gif_name", default)
    value = str(value).strip() if value is not None else ""
    return value or default


def resolve_gif_fps(cfg, default=30):
    run_cfg = _run_cfg(cfg)
    raw = run_cfg.get("gif_fps", run_cfg.get("fps", default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def load_ursina_loop():
    try:
        from visuals.visual_ursina import run_ursina_loop

        return run_ursina_loop
    except ModuleNotFoundError as exc:
        missing_name = str(getattr(exc, "name", "") or "")
        if missing_name == "ursina" or missing_name.startswith("ursina."):
            raise RuntimeError(
                "3D renderer requested, but 'ursina' is not installed. "
                "Install it with `pip install ursina` or `pip install -r requirements.txt`."
            ) from exc
        raise
