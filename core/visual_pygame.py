import numpy as np
import pygame


def _project_to_screen_px(env, pos, width, height):
    # Ортографическая проекция 3D на 2D экран
    sx = pos[0] / (env.world_width / 2)
    sz = pos[2] / env.world_height
    px = int((width / 2) + sx * (width / 2))
    py = int(height - sz * height)
    return px, py


def _asteroid_screen_radius(env, pos, width, height, depth_boost=1.2):
    px_per_world = min(width / env.world_width, height / env.world_height)
    base_px = env.asteroid_radius * px_per_world
    depth_ratio = np.clip(pos[1] / env.spawn_y, 0.0, 1.0)
    scale = 1.0 + depth_boost * (1.0 - depth_ratio)
    return max(1, int(base_px * scale))


def _projectile_screen_radius(env, pos, base_px=3, scale_px=8):
    depth_ratio = np.clip(pos[1] / env.spawn_y, 0.0, 1.0)
    return int(base_px + scale_px * (1.0 - depth_ratio))


def _aim_point_radius(env, pos, width, height):
    px_per_world = min(width / env.world_width, height / env.world_height)
    base_px = max(2.0, 0.6 * env.asteroid_radius * px_per_world)
    depth_ratio = np.clip(pos[1] / env.spawn_y, 0.0, 1.0)
    return int(base_px + 6.0 * (1.0 - depth_ratio))


def _closest_point_on_segment(start, end, point):
    s = np.array(start, dtype=np.float32)
    e = np.array(end, dtype=np.float32)
    p = np.array(point, dtype=np.float32)
    d = e - s
    denom = float(np.dot(d, d))
    if denom <= 1e-6:
        return s, 0.0
    t = float(np.dot(p - s, d) / denom)
    t = max(0.0, min(1.0, t))
    return s + d * t, t


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


def _draw_gun_body(screen, width, height, base_w=60, base_h=40):
    center = (width // 2, height)
    base_rect = pygame.Rect(0, 0, base_w, base_h)
    base_rect.midbottom = center
    pygame.draw.rect(screen, (40, 80, 200), base_rect)
    return center, base_w


def _gun_line_end(env, width, height):
    direction = env._gun_direction()
    end_3d = direction * env.projectile_max_dist
    return _project_to_screen_px(env, end_3d, width, height)


def _compute_lines(end, width, height, base_w):
    center = (width // 2, height)
    half = base_w // 2
    return [
        (center, end, (220, 40, 40), 3),  # main red
        ((center[0] - half, center[1]), end, (40, 200, 40), 2),  # left green
        ((center[0] + half, center[1]), end, (40, 200, 40), 2),  # right green
    ]


def _occlusion_indices(lines, asteroids):
    redraw_indices = set()
    for start, end, color, width in lines:
        for idx, ast in enumerate(asteroids):
            c = ast["screen"]
            r = ast["radius"]
            closest, _ = _closest_point_on_segment(start, end, c)
            dist = float(np.linalg.norm(closest - np.array(c, dtype=np.float32)))
            if dist <= r * 1.2 and closest[1] > c[1]:
                redraw_indices.add(idx)
    return redraw_indices


class PygameRenderer:
    def __init__(self, width=800, height=600, title="Asteroid Defense"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("Arial", 16)
        self.clock = pygame.time.Clock()
        self.show_dead_zone = False
        self._dead_zone_cache = None
        self.last_events = []

    def process_events(self):
        events = pygame.event.get()
        self.last_events = events
        for event in events:
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_z:
                self.show_dead_zone = not self.show_dead_zone
        return True

    def _compute_dead_zone(self, env):
        # Compute pitch_total range and yaw range based on current config
        half_fov = env.fov / 2.0
        base_pitch = env.base_pitch
        pitch_min = max(0.0, base_pitch - half_fov)
        pitch_max = min(np.pi / 2.0, base_pitch + half_fov)
        yaw_min = -half_fov
        yaw_max = half_fov
        self._dead_zone_cache = (pitch_min, pitch_max, yaw_min, yaw_max)
        return self._dead_zone_cache

    def _is_in_dead_zone(self, env, pos):
        if self._dead_zone_cache is None:
            self._compute_dead_zone(env)
        pitch_min, pitch_max, yaw_min, yaw_max = self._dead_zone_cache
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        if y <= 1e-6:
            return True
        yaw = np.arctan2(x, y)
        pitch_total = np.arctan2(z, y)
        return (yaw < yaw_min) or (yaw > yaw_max) or (pitch_total < pitch_min) or (pitch_total > pitch_max)

    def draw(self, env, reward=None, total_reward=None, extra_lines=None, fps=30):
        self.screen.fill((15, 15, 20))

        # Prepare asteroid draw data
        asteroids_screen = []
        for ast in env.asteroids:
            x, y = _project_to_screen_px(env, ast["pos"], self.width, self.height)
            r = _asteroid_screen_radius(env, ast["pos"], self.width, self.height)
            color_intensity = int(255 * (1.0 - np.clip(ast["pos"][1] / env.spawn_y, 0, 0.8)))
            if self.show_dead_zone and self._is_in_dead_zone(env, ast["pos"]):
                color = (220, 60, 60)
            else:
                color = (color_intensity, color_intensity, color_intensity)
            asteroids_screen.append({
                "screen": (x, y),
                "radius": r,
                "color": color,
                "pos": ast["pos"],
            })

        # Draw asteroids first pass
        for ast in asteroids_screen:
            pygame.draw.circle(self.screen, ast["color"], ast["screen"], ast["radius"])

        # Draw gun body
        _, base_w = _draw_gun_body(self.screen, self.width, self.height)

        # 3D ray hit for clipping/aim point
        ray_origin = np.zeros(3, dtype=np.float32)
        ray_dir = env._gun_direction()
        hit_t = None
        for ast in asteroids_screen:
            t = _ray_sphere_hit(ray_origin, ray_dir, ast["pos"], env.asteroid_radius)
            if t is not None and (hit_t is None or t < hit_t):
                hit_t = t

        if hit_t is not None:
            hit_point = ray_dir * hit_t
            line_end = _project_to_screen_px(env, hit_point, self.width, self.height)
        else:
            hit_point = None
            line_end = _gun_line_end(env, self.width, self.height)

        lines = _compute_lines(line_end, self.width, self.height, base_w)
        redraw_indices = _occlusion_indices(lines, asteroids_screen)

        main_tip = None
        for i, (start, end, color, lw) in enumerate(lines):
            pygame.draw.line(self.screen, color, start, end, lw)
            if i == 0:
                main_tip = end

        # Aim point with occlusion
        aim_drawn = False
        aim_pos = None
        if main_tip is not None:
            if hit_point is not None:
                aim_pos = hit_point
            else:
                aim_pos = ray_dir * env.projectile_max_dist
            aim_r = _aim_point_radius(env, aim_pos, self.width, self.height)

            aim_under = False
            for ast in asteroids_screen:
                c = ast["screen"]
                r = ast["radius"]
                dist = float(np.linalg.norm(np.array(main_tip, dtype=np.float32) - np.array(c, dtype=np.float32)))
                if dist <= r and main_tip[1] > c[1] and hit_point is None:
                    aim_under = True
                    break

            if aim_under:
                pygame.draw.circle(self.screen, (255, 80, 80), main_tip, aim_r)
                aim_drawn = True

        # Redraw asteroids that should be in front
        for idx in sorted(redraw_indices):
            ast = asteroids_screen[idx]
            pygame.draw.circle(self.screen, ast["color"], ast["screen"], ast["radius"])

        if main_tip is not None and not aim_drawn:
            aim_r = _aim_point_radius(env, aim_pos, self.width, self.height)
            pygame.draw.circle(self.screen, (255, 80, 80), main_tip, aim_r)

        # Draw projectiles
        for p in env.projectiles:
            x, y = _project_to_screen_px(env, p["pos"], self.width, self.height)
            r = _projectile_screen_radius(env, p["pos"])
            pygame.draw.circle(self.screen, (255, 255, 60), (x, y), r)

        # HUD
        y = 10
        if reward is not None and total_reward is not None:
            text = self.font.render(
                f"Step Reward: {reward:+.2f} | Total: {total_reward:+.2f}", True, (255, 200, 100)
            )
            self.screen.blit(text, (10, y))
            y += 20

        text2 = self.font.render(
            f"Kills: {env.kills} | Hull Damage: {env.hull_damage}", True, (255, 200, 100)
        )
        self.screen.blit(text2, (10, y))
        y += 20

        if extra_lines:
            for line in extra_lines:
                text = self.font.render(line, True, (230, 230, 230))
                self.screen.blit(text, (10, y))
                y += 18

        pygame.display.flip()
        self.clock.tick(fps)

    def close(self):
        pygame.quit()
