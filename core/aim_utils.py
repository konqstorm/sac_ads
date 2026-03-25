import numpy as np


AIM_SLOT_DIM = 5


def extract_aim_obs(full_obs, slot_idx, slot_dim=AIM_SLOT_DIM):
    """
    Возвращает observation для наводчика:
    [slot_features(5), yaw_norm, pitch_norm] -> 7 dims.
    """
    full_obs = np.asarray(full_obs, dtype=np.float32)
    if full_obs.ndim != 1:
        full_obs = full_obs.flatten()

    yaw_pitch = full_obs[-2:] if full_obs.shape[0] >= 2 else np.zeros(2, dtype=np.float32)

    start = int(slot_idx) * slot_dim if slot_idx is not None else -1
    end = start + slot_dim
    if start < 0 or end > (full_obs.shape[0] - 2):
        slot = np.zeros(slot_dim, dtype=np.float32)
    else:
        slot = full_obs[start:end]

    return np.concatenate([slot, yaw_pitch]).astype(np.float32)
