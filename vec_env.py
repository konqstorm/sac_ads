import numpy as np
from env import AsteroidDefenseEnv


class VecEnv:
    """
    Простая векторизованная обёртка: несколько AsteroidDefenseEnv
    работают последовательно в одном процессе, но данные из всех
    попадают в один replay buffer — это уже даёт хороший прирост
    разнообразия опыта и эффективного использования CPU.
    """

    def __init__(self, env_config: dict, n_envs: int, seeds: list[int] | None = None):
        self.n_envs = n_envs
        self.envs = [AsteroidDefenseEnv(env_config) for _ in range(n_envs)]
        if seeds is None:
            seeds = [i * 100 for i in range(n_envs)]
        self._seeds = seeds
        self._obs = [None] * n_envs

    # ------------------------------------------------------------------
    # Curriculum: обновить параметры всех сред разом
    # ------------------------------------------------------------------
    def apply_stage(self, stage: dict):
        skip = {"until_ep"}
        for env in self.envs:
            for key, val in stage.items():
                if key not in skip and hasattr(env, key):
                    setattr(env, key, val)

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(self) -> list[np.ndarray]:
        self._obs = [
            env.reset(seed=self._seeds[i])[0]
            for i, env in enumerate(self.envs)
        ]
        return list(self._obs)

    def step(self, actions: list[np.ndarray]):
        """
        Возвращает lists: obs2, rewards, dones, infos.
        Автоматически сбрасывает среды, которые завершились.
        """
        obs2_list, reward_list, done_list, info_list = [], [], [], []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs2, reward, done, _, info = env.step(action)
            if done:
                obs2, _ = env.reset()
            obs2_list.append(obs2)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        self._obs = obs2_list
        return obs2_list, reward_list, done_list, info_list

    @property
    def obs(self) -> list[np.ndarray]:
        return self._obs

    # Удобный доступ к пространствам действий/наблюдений первой среды
    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space
