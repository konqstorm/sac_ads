import os
import yaml
import torch
import numpy as np

from core.env import AsteroidDefenseEnv
from core.models_discrete import Actor, Critic
from core.sac_discrete import SAC
from core.baseline_mode import BaselineController
from core.aimer import FrozenAimer
from core.two_stage_agent import TwoStageAgent
from core.aim_utils import extract_aim_obs


def _load_env(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return AsteroidDefenseEnv(cfg["env"]), cfg


def _load_agent(env, cfg, weights_dir=None):
    obs_dim = env.observation_space.shape[0]
    act_dim = int(cfg.get("env", {}).get("max_asteroids", 5))
    actor = Actor(obs_dim, act_dim)
    critic1 = Critic(obs_dim, act_dim)
    critic2 = Critic(obs_dim, act_dim)
    agent = SAC(actor, critic1, critic2, cfg.get("agent", {}))
    if weights_dir is None:
        weights_dir = cfg.get("train", {}).get("save_dir_selector", "results/weights_selector")
    actor.load_state_dict(torch.load(os.path.join(weights_dir, "actor.pt"), map_location="cpu"))
    c1 = os.path.join(weights_dir, "critic1.pt")
    c2 = os.path.join(weights_dir, "critic2.pt")
    if os.path.exists(c1):
        critic1.load_state_dict(torch.load(c1, map_location="cpu"))
    if os.path.exists(c2):
        critic2.load_state_dict(torch.load(c2, map_location="cpu"))
    actor.eval(); critic1.eval(); critic2.eval()
    return agent


def _select_best_slot(env):
    best_idx = None
    best_score = None
    for i, a in enumerate(env.asteroid_slots):
        if a is None:
            continue
        r = a["pos"].astype(np.float32)
        v = a["vel"].astype(np.float32)
        t = env._solve_intercept(r, v, env.projectile_speed)
        aim_point = r + v * t if t is not None else r
        target_yaw, target_pitch = env._direction_to_yaw_pitch(aim_point)
        err_yaw = target_yaw - env.yaw
        err_pitch = target_pitch - env.pitch
        score = abs(err_yaw) + abs(err_pitch)
        if best_idx is None or score < best_score:
            best_idx = i
            best_score = score
    return best_idx


def run_eval(mode="baseline", cfg_path=os.path.join("configs", "config_select.yaml"), episodes=100):
    env, cfg = _load_env(cfg_path)

    if mode == "baseline":
        controller = BaselineController(env)
    elif mode == "agent":
        agent = _load_agent(env, cfg)
        aimer_cfg = cfg.get("aimer", {})
        aim_obs_dim = aimer_cfg.get("obs_dim", 7)
        aim_act_dim = aimer_cfg.get("action_dim", 3)
        aim_weights_dir = aimer_cfg.get("weights_dir", "results/weights_aim")
        aim_det = bool(aimer_cfg.get("deterministic_eval", True))
        aimer = FrozenAimer(aim_obs_dim, aim_act_dim, aim_weights_dir)
        commit_steps = cfg.get("agent", {}).get("commit_steps", 15)
        two_stage = TwoStageAgent(
            selector=agent,
            aimer=aimer,
            fire_threshold=env.fire_threshold,
            commit_steps=commit_steps,
            deterministic_selector=True,
            deterministic_aimer=aim_det,
        )
        controller = None
    elif mode == "aimer":
        aim_cfg = cfg.get("aim", {})
        aimer_cfg = cfg.get("aimer", {})
        aim_obs_dim = aim_cfg.get("obs_dim", 7)
        aim_act_dim = aim_cfg.get("action_dim", 3)
        aim_weights_dir = aimer_cfg.get("weights_dir", "results/weights_aim")
        aim_det = bool(aimer_cfg.get("deterministic_eval", True))
        aimer = FrozenAimer(aim_obs_dim, aim_act_dim, aim_weights_dir)
        controller = None
    else:
        raise SystemExit("mode must be: baseline | best_baseline | agent | aimer")

    seed_base = cfg.get("train", {}).get("seed", 42)

    total_kills = 0
    total_hull = 0
    wins = 0

    for ep in range(episodes):
        ep_seed = seed_base + ep
        env.reset(seed=ep_seed)
        if controller is not None:
            controller = controller.__class__(env)
        if mode == "agent":
            two_stage.reset()

        done = False
        while not done:
            if mode in ("baseline", "best_baseline"):
                action = controller.act()
            elif mode == "agent":
                obs = env._get_obs()
                action, _, _, _ = two_stage.step(obs, env)
            else:  # aimer
                obs = env._get_obs()
                target_idx = _select_best_slot(env)
                if target_idx is None:
                    action = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                else:
                    aim_obs = extract_aim_obs(obs, target_idx)
                    action = aimer.act(aim_obs, deterministic=aim_det).astype(np.float32)
            _, _, done, _, _ = env.step(action)

        win = bool(env.hp > 0 and env.asteroids_remaining == 0 and len(env.asteroids) == 0)
        total_kills += env.kills
        total_hull += env.hull_damage
        if win:
            wins += 1

        print(f"seed={ep_seed} | kills={env.kills} | hull={env.hull_damage} | win={win}")

    print(f"mode={mode} episodes={episodes}")
    print(f"kills_total={total_kills} hull_damage_total={total_hull} winrate={wins/episodes:.2f}")


if __name__ == "__main__":
    run_eval(mode="baseline")
