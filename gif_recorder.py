import os
import atexit

import numpy as np
from PIL import Image


class GIFRecorder:
    def __init__(self, enabled=False, directory="tmp_gif", name="run.gif", fps=30):
        self.enabled = bool(enabled)
        self.directory = str(directory or "tmp_gif")
        self.name = str(name or "run.gif")
        self.fps = max(1, int(fps))
        self.frames = []
        self._saved = False

        if not self.name.lower().endswith(".gif"):
            self.name += ".gif"

        if self.enabled:
            os.makedirs(self.directory, exist_ok=True)
            print(f"[gif] enabled -> {self.output_path}")
            atexit.register(self._save_on_exit)

    @property
    def output_path(self):
        return os.path.join(self.directory, self.name)

    def add_frame(self, frame_rgb):
        if not self.enabled or frame_rgb is None:
            return
        frame = np.asarray(frame_rgb, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            return
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        self.frames.append(Image.fromarray(frame, mode="RGB"))

    def save(self):
        if not self.enabled or self._saved:
            return None
        self._saved = True
        if not self.frames:
            print(f"[gif] no frames captured -> {self.output_path}")
            return None

        os.makedirs(self.directory, exist_ok=True)
        duration_ms = max(1, int(round(1000 / self.fps)))
        first, *rest = self.frames
        first.save(
            self.output_path,
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        print(f"[gif] saved {len(self.frames)} frames -> {self.output_path}")
        self.frames.clear()
        return self.output_path

    def _save_on_exit(self):
        if self.enabled and not self._saved:
            self.save()
