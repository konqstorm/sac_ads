from typing import Callable, Iterable, Optional

import numpy as np
from ursina import (
    AmbientLight,
    DirectionalLight,
    Entity,
    Text,
    Ursina,
    Vec3,
    application,
    camera,
    clamp,
    color,
    destroy,
    held_keys,
    mouse,
    time,
    window,
)
from ursina.shaders import unlit_shader
from panda3d.core import PNMImage


def _c(r, g, b, a=255):
    # Ursina 8.3.0's color.rgb/rgba helpers may not normalize 0..255 values.
    # Convert explicit 8-bit colors to normalized floats to avoid white-clamped rendering.
    return color.rgba(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, float(a) / 255.0)


def _ray_sphere_hit(origin, direction, center, radius):
    o = np.array(origin, dtype=np.float32)
    d = np.array(direction, dtype=np.float32)
    c = np.array(center, dtype=np.float32)
    oc = o - c
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(oc, d))
    c_val = float(np.dot(oc, oc) - radius * radius)
    disc = b * b - 4.0 * a * c_val
    if disc < 0.0 or a <= 1e-6:
        return None
    sqrt_disc = float(np.sqrt(disc))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    candidates = [t for t in (t1, t2) if t > 1e-4]
    return min(candidates) if candidates else None


class UrsinaRenderer:
    def __init__(self, env, width=1280, height=720, title="Asteroid Defense 3D", fps=30, gif_recorder=None):
        self.env = env
        self.fps = fps
        self.gif_recorder = gif_recorder
        self.show_dead_zone = False
        self._dead_zone_cache = None
        # Compress long depth ranges so asteroids stay visible in the free camera view.
        self.world_scale = min(1.0, 40.0 / max(40.0, float(env.projectile_max_dist)))

        self.app = Ursina(
            borderless=False,
            development_mode=False,
            editor_ui_enabled=False,
            title=title,
        )
        window.title = title
        window.size = (width, height)
        window.color = _c(7, 10, 16)
        camera.shader = None
        camera.clear_color = _c(7, 10, 16)
        camera.overlay.color = _c(0, 0, 0, 0)
        camera.overlay.enabled = False
        if hasattr(window, "render_mode"):
            window.render_mode = "default"
        if hasattr(window, "fps_counter"):
            window.fps_counter.enabled = False
        if hasattr(window, "entity_counter"):
            window.entity_counter.enabled = False
        if hasattr(window, "collider_counter"):
            window.collider_counter.enabled = False
        if hasattr(window, "exit_button"):
            window.exit_button.enabled = False
        application.target_frame_rate = int(max(1, fps))
        camera.fov = 90
        camera.clip_plane_near = 0.05
        camera.clip_plane_far = max(400.0, float(env.spawn_y) * 5.0 * self.world_scale)

        DirectionalLight(rotation=(35, -40, 20), intensity=1.2, color=_c(220, 220, 220))
        DirectionalLight(rotation=(-20, 35, -10), intensity=0.7, color=_c(170, 170, 200))
        AmbientLight(color=_c(120, 120, 150, 200))

        floor_scale = (
            max(float(env.world_width), float(env.spawn_y), float(env.world_height))
            * self.world_scale
            * 1.8
        )
        Entity(
            model="plane",
            scale=floor_scale,
            y=-0.08,
            color=_c(18, 22, 34),
            shader=unlit_shader,
        )

        spawn_z_center = float(env.spawn_z_center or (env.spawn_z_min + env.spawn_z_max) / 2.0)
        camera_start_y = max(4.0, spawn_z_center * self.world_scale)
        camera_start_z = -12.0
        self.fly_camera = _FlyCamera(
            position=(0.0, camera_start_y, camera_start_z),
            yaw=0.0,
            pitch=8.0,
            move_speed=max(8.0, float(env.spawn_y) * 0.06 * self.world_scale),
            mouse_sensitivity=60.0,
        )

        asteroid_radius_world = float(env.asteroid_radius) * self.world_scale
        base_size = max(0.8, asteroid_radius_world * 0.8)
        self.cannon_base = Entity(
            model="cube",
            scale=(base_size * 2.0, base_size * 0.9, base_size * 1.8),
            position=(0.0, base_size * 0.45, 0.0),
            color=_c(45, 75, 180),
            shader=unlit_shader,
        )
        self.cannon_barrel = Entity(
            model="cube",
            color=_c(210, 65, 65),
            origin_z=-0.5,
            scale=(base_size * 0.25, base_size * 0.25, max(1.5, env.projectile_max_dist * 0.08 * self.world_scale)),
            position=(0.0, self.cannon_base.y + base_size * 0.3, 0.0),
            shader=unlit_shader,
        )

        self.aim_ray = Entity(
            model="cube",
            color=_c(255, 90, 90, 130),
            origin_z=-0.5,
            scale=(0.08, 0.08, max(1.0, env.projectile_max_dist * 0.2 * self.world_scale)),
            position=(0.0, self.cannon_barrel.y, 0.0),
            shader=unlit_shader,
        )
        self.hit_marker = Entity(
            model="sphere",
            scale=max(0.2, asteroid_radius_world * 0.35),
            color=_c(255, 85, 85),
            enabled=False,
            shader=unlit_shader,
        )

        self.asteroid_entities = {}
        self.projectile_entities = []

        self.hud = Text(
            text="Initializing 3D renderer...",
            origin=(-0.5, -0.5),
            position=(-0.49, -0.48),
            scale=1.05,
            background=False,
            color=_c(250, 220, 150),
        )
        self._clear_color = _c(7, 10, 16)
        self._gif_warning_printed = False

    def toggle_dead_zone(self):
        self.show_dead_zone = not self.show_dead_zone

    def _compute_dead_zone(self):
        half_fov = self.env.fov / 2.0
        base_pitch = self.env.base_pitch
        pitch_min = max(0.0, base_pitch - half_fov)
        pitch_max = min(np.pi / 2.0, base_pitch + half_fov)
        yaw_min = -half_fov
        yaw_max = half_fov
        self._dead_zone_cache = (pitch_min, pitch_max, yaw_min, yaw_max)
        return self._dead_zone_cache

    def _is_in_dead_zone(self, pos):
        if self._dead_zone_cache is None:
            self._compute_dead_zone()
        pitch_min, pitch_max, yaw_min, yaw_max = self._dead_zone_cache
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        if y <= 1e-6:
            return True
        yaw = np.arctan2(x, y)
        pitch_total = np.arctan2(z, y)
        return (yaw < yaw_min) or (yaw > yaw_max) or (pitch_total < pitch_min) or (pitch_total > pitch_max)

    def _sync_cannon(self):
        direction_env = self.env._gun_direction()
        direction_world = self._env_to_world(direction_env)
        origin = Vec3(0.0, self.cannon_barrel.y, 0.0)
        tip = origin + Vec3(*direction_world) * max(2.0, self.env.projectile_max_dist * 0.08)
        self.cannon_barrel.position = origin
        self.cannon_barrel.look_at(tip)

    def _sync_asteroids(self):
        alive_ids = set()
        for asteroid in self.env.asteroids:
            asteroid_id = int(asteroid["id"])
            alive_ids.add(asteroid_id)
            entity = self.asteroid_entities.get(asteroid_id)
            if entity is None:
                entity = Entity(
                    model="sphere",
                    scale=self.env.asteroid_radius * 2.0 * self.world_scale,
                    color=color.light_gray,
                    shader=unlit_shader,
                )
                self.asteroid_entities[asteroid_id] = entity

            depth_ratio = float(np.clip(asteroid["pos"][1] / self.env.spawn_y, 0.0, 1.0))
            if self.show_dead_zone and self._is_in_dead_zone(asteroid["pos"]):
                entity.color = _c(230, 70, 70)
            else:
                intensity = int(95 + (1.0 - depth_ratio) * 160)
                entity.color = _c(intensity, intensity, intensity)

            entity.position = Vec3(*self._env_to_world(asteroid["pos"]))
            entity.enabled = True

        for asteroid_id in list(self.asteroid_entities.keys()):
            if asteroid_id not in alive_ids:
                destroy(self.asteroid_entities[asteroid_id])
                del self.asteroid_entities[asteroid_id]

    def _sync_projectiles(self):
        needed = len(self.env.projectiles)
        while len(self.projectile_entities) < needed:
            self.projectile_entities.append(
                Entity(
                    model="sphere",
                    scale=max(0.08, self.env.asteroid_radius * 0.25 * self.world_scale),
                    color=_c(255, 240, 90),
                    shader=unlit_shader,
                )
            )

        for idx, entity in enumerate(self.projectile_entities):
            if idx < needed:
                projectile = self.env.projectiles[idx]
                entity.position = Vec3(*self._env_to_world(projectile["pos"]))
                entity.enabled = True
            else:
                entity.enabled = False

    def _sync_aim_helpers(self):
        ray_origin = np.zeros(3, dtype=np.float32)
        ray_dir = self.env._gun_direction()
        hit_t = None
        for asteroid in self.env.asteroids:
            t_val = _ray_sphere_hit(ray_origin, ray_dir, asteroid["pos"], self.env.asteroid_radius)
            if t_val is not None and (hit_t is None or t_val < hit_t):
                hit_t = t_val

        distance = float(self.env.projectile_max_dist if hit_t is None else hit_t) * self.world_scale
        distance = max(0.01, distance)
        origin_world = Vec3(0.0, self.cannon_barrel.y, 0.0)
        dir_world = Vec3(*self._env_to_world(ray_dir)).normalized()
        tip_world = origin_world + dir_world * distance

        self.aim_ray.position = origin_world
        self.aim_ray.look_at(tip_world)
        self.aim_ray.scale = (0.08, 0.08, distance)

        self.hit_marker.enabled = hit_t is not None
        if self.hit_marker.enabled:
            self.hit_marker.position = tip_world

    def draw(self, reward=None, total_reward=None, extra_lines: Optional[Iterable[str]] = None):
        # Guard against runtime/editor settings forcing white viewport.
        window.color = self._clear_color
        camera.clear_color = self._clear_color
        camera.overlay.enabled = False
        if hasattr(window, "render_mode"):
            window.render_mode = "default"

        self.fly_camera.step()
        self._sync_cannon()
        self._sync_asteroids()
        self._sync_projectiles()
        self._sync_aim_helpers()

        lines = []
        if reward is not None and total_reward is not None:
            lines.append(f"Step Reward: {reward:+.2f} | Total: {total_reward:+.2f}")
        lines.append(f"Kills: {self.env.kills} | Hull Damage: {self.env.hull_damage}")
        lines.append(f"Dead Zone: {'ON' if self.show_dead_zone else 'OFF'} (Z to toggle)")
        if extra_lines:
            lines.extend([str(line) for line in extra_lines])
        self.hud.text = "\n".join(lines)
        self._capture_frame()

    def _capture_frame(self):
        if self.gif_recorder is None:
            return
        try:
            win = application.base.win

            # Primary path: capture as Texture (works reliably on many Panda3D backends).
            tex = win.getScreenshot()
            if tex is not None and tex.hasRamImage():
                width = tex.getXSize()
                height = tex.getYSize()
                raw = bytes(tex.getRamImageAs("RGB"))
                if raw:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                    frame = np.flipud(frame)
                    self.gif_recorder.add_frame(frame)
                    return

            # Fallback path: capture as PNMImage.
            pnm = PNMImage()
            ok = win.getScreenshot(pnm)
            if ok:
                width = pnm.getXSize()
                height = pnm.getYSize()
                raw = bytes(pnm.getRamImageAs("RGB"))
                if raw:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                    frame = np.flipud(frame)
                    self.gif_recorder.add_frame(frame)
        except Exception as exc:
            if not self._gif_warning_printed:
                print(f"[gif] warning: could not capture Ursina frame: {exc}")
                self._gif_warning_printed = True

    def _finalize(self):
        if self.gif_recorder is not None:
            self.gif_recorder.save()
        mouse.locked = False
        mouse.visible = True

    def close(self):
        self._finalize()
        application.quit()

    def _env_to_world(self, pos):
        # Env: (x, y=depth, z=height) -> Ursina: (x, y=height, z=depth), with uniform scale.
        return np.array(
            [
                float(pos[0]) * self.world_scale,
                float(pos[2]) * self.world_scale,
                float(pos[1]) * self.world_scale,
            ],
            dtype=np.float32,
        )


def _manual_action():
    yaw_action = float(held_keys["right arrow"] - held_keys["left arrow"])
    pitch_action = float(held_keys["up arrow"] - held_keys["down arrow"])
    fire_action = 1.0 if held_keys["space"] else -1.0
    return np.array([yaw_action, pitch_action, fire_action], dtype=np.float32)


class _FlyCamera:
    def __init__(self, position, yaw=0.0, pitch=0.0, move_speed=8.0, mouse_sensitivity=60.0):
        self.rig = Entity(position=position)
        camera.parent = self.rig
        camera.position = Vec3(0.0, 0.0, 0.0)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.move_speed = float(move_speed)
        self.mouse_sensitivity = float(mouse_sensitivity)
        self.rig.rotation_y = self.yaw
        camera.rotation_x = self.pitch
        mouse.locked = True
        mouse.visible = False

    def step(self):
        self.yaw += float(mouse.velocity[0]) * self.mouse_sensitivity
        self.pitch -= float(mouse.velocity[1]) * self.mouse_sensitivity
        self.pitch = clamp(self.pitch, -89.0, 89.0)
        self.rig.rotation_y = self.yaw
        camera.rotation_x = self.pitch

        forward = self.rig.forward * (held_keys["w"] - held_keys["s"])
        right = self.rig.right * (held_keys["d"] - held_keys["a"])
        vertical = Vec3(0.0, 1.0, 0.0) * (held_keys["e"] - held_keys["q"])
        direction = (forward + right + vertical)
        if direction.length() > 0:
            direction = direction.normalized()
        self.rig.position += direction * self.move_speed * time.dt


class _SimulationLoop(Entity):
    def __init__(
        self,
        env,
        renderer,
        action_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        initial_obs=None,
        on_episode_reset: Optional[Callable[[], None]] = None,
        extra_lines_fn: Optional[Callable[[np.ndarray], Optional[Iterable[str]]]] = None,
        manual_controls=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.env = env
        self.renderer = renderer
        self.action_fn = action_fn
        self.on_episode_reset = on_episode_reset
        self.extra_lines_fn = extra_lines_fn
        self.manual_controls = manual_controls

        if initial_obs is None:
            self.obs, _ = self.env.reset()
        else:
            self.obs = initial_obs

        self.total_reward = 0.0
        self.renderer.draw(reward=None, total_reward=None, extra_lines=self._extra_lines())

    def _extra_lines(self):
        if self.extra_lines_fn is None:
            return None
        return self.extra_lines_fn(self.obs)

    def _next_action(self):
        if self.manual_controls:
            return _manual_action()
        if self.action_fn is None:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        action = self.action_fn(self.obs)
        return np.asarray(action, dtype=np.float32)

    def input(self, key):
        if key == "escape":
            self.renderer.close()
            return
        if key == "z":
            self.renderer.toggle_dead_zone()

    def update(self):
        action = self._next_action()
        self.obs, reward, done, _, _ = self.env.step(action)
        self.total_reward += reward

        self.renderer.draw(
            reward=reward,
            total_reward=self.total_reward,
            extra_lines=self._extra_lines(),
        )

        if done:
            if self.on_episode_reset is not None:
                maybe_obs = self.on_episode_reset()
                if maybe_obs is not None:
                    self.obs = maybe_obs
                else:
                    self.obs, _ = self.env.reset()
            else:
                self.obs, _ = self.env.reset()
            self.total_reward = 0.0


def run_ursina_loop(
    env,
    title="Asteroid Defense 3D",
    fps=30,
    action_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    initial_obs=None,
    on_episode_reset: Optional[Callable[[], None]] = None,
    extra_lines_fn: Optional[Callable[[np.ndarray], Optional[Iterable[str]]]] = None,
    manual_controls=False,
    gif_recorder=None,
):
    renderer = UrsinaRenderer(env=env, title=title, fps=fps, gif_recorder=gif_recorder)
    _SimulationLoop(
        env=env,
        renderer=renderer,
        action_fn=action_fn,
        initial_obs=initial_obs,
        on_episode_reset=on_episode_reset,
        extra_lines_fn=extra_lines_fn,
        manual_controls=manual_controls,
    )
    renderer.app.run()
    renderer._finalize()
