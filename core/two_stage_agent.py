import numpy as np

from core.aim_utils import extract_aim_obs


class TwoStageAgent:
    """
    Агент выбора цели + замороженный наводчик.
    Управляет коммитом к цели и выдаёт action для среды.
    """
    def __init__(
        self,
        selector,
        aimer,
        fire_threshold,
        commit_steps=None,
        sample_every=None,
        deterministic_selector=True,
        deterministic_aimer=True,
    ):
        self.selector = selector
        self.aimer = aimer
        self.fire_threshold = float(fire_threshold)
        self.commit_steps = commit_steps
        self.sample_every = sample_every
        self.deterministic_selector = deterministic_selector
        self.deterministic_aimer = deterministic_aimer
        self.reset()

    def reset(self):
        self.current_target = None
        self.steps_on_target = 0
        self.steps_since_sample = 0
        self.fired_since_select = False

    def _select_slot(self, env, raw_action):
        n = len(env.asteroid_slots)
        if n <= 0:
            return None
        try:
            idx = int(raw_action)
        except Exception:
            return None
        if idx < 0 or idx >= n:
            return None
        if env.asteroid_slots[idx] is None:
            return None
        return idx

    def step(self, obs, env):
        if self.deterministic_selector:
            raw_action = self.selector.act_deterministic(obs)
        else:
            raw_action = self.selector.act(obs)

        if self.sample_every is not None:
            if self.steps_since_sample == 0 or self.steps_since_sample >= self.sample_every:
                self.current_target = self._select_slot(env, raw_action)
                self.steps_since_sample = 0
        else:
            if self.commit_steps is None:
                self.commit_steps = 1

            target_lost = True
            if self.current_target is not None:
                if 0 <= self.current_target < len(env.asteroid_slots):
                    if env.asteroid_slots[self.current_target] is not None:
                        target_lost = False
            if target_lost or self.steps_on_target >= self.commit_steps:
                self.current_target = self._select_slot(env, raw_action)
                self.steps_on_target = 0
                self.fired_since_select = False

        if self.current_target is None:
            action = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            fired = False
        else:
            aim_obs = extract_aim_obs(obs, self.current_target)
            action = self.aimer.act(aim_obs, deterministic=self.deterministic_aimer).astype(np.float32)
            fired = bool(action[2] > self.fire_threshold)

        if self.sample_every is not None:
            if fired:
                self.steps_since_sample = self.sample_every
            self.steps_since_sample += 1
        else:
            if fired:
                self.steps_on_target = self.commit_steps
                self.fired_since_select = True
            else:
                self.steps_on_target += 1

        return action, raw_action, self.current_target, fired
