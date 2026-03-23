import sys
import yaml
import numpy as np
import pygame

from env import AsteroidDefenseEnv
#from train import train_agent

def _load_env(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return AsteroidDefenseEnv(cfg["env"])

def _project_to_screen_px(env, pos, width, height):
    # Ортографическая проекция 3D на 2D экран
    # X-ось мапится от -world_width/2 до +world_width/2
    # Z-ось мапится от 0 до world_height
    # У PyGame Y идет вниз, поэтому мы вычитаем высоту
    sx = pos[0] / (env.world_width / 2)
    sz = pos[2] / env.world_height
    
    px = int((width / 2) + sx * (width / 2))
    py = int(height - sz * height)
    return px, py

def _asteroid_screen_radius(env, pos, base_px=4, scale_px=20):
    # Чем ближе астероид по Y, тем больше его радиус
    depth_ratio = np.clip(pos[1] / env.spawn_y, 0.0, 1.0)
    return int(base_px + scale_px * (1.0 - depth_ratio))

def _draw_gun(screen, env, width, height):
    # Пушка всегда снизу по центру (0,0,0)
    center = (width // 2, height)
    
    # Считаем, куда направлен прицел
    direction = env._gun_direction()
    # Проецируем линию длиной 150 единиц
    end_3d = direction * env.projectile_max_dist
    end_x, end_y = _project_to_screen_px(env, end_3d, width, height)

    # Рисуем базу (квадратик)
    base_rect = pygame.Rect(0, 0, 60, 40)
    base_rect.midbottom = center
    pygame.draw.rect(screen, (40, 80, 200), base_rect)

    # Рисуем линию прицела (красная)
    pygame.draw.line(screen, (220, 40, 40), center, (end_x, end_y), 3)

def manual_mode(cfg_path="config.yaml"):
    env = _load_env(cfg_path)
    obs, _ = env.reset()

    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Asteroid Defense - Manual")

    font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()

    running = True
    total_reward = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        yaw_action, pitch_action, fire_action = 0.0, 0.0, -1.0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            yaw_action -= 0.25
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            yaw_action += 0.25
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            pitch_action += 0.25
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            pitch_action -= 0.25
        if keys[pygame.K_SPACE]:
            fire_action = 0.25

        action = np.array([yaw_action, pitch_action, fire_action], dtype=np.float32)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        screen.fill((15, 15, 20))

        # Отрисовка астероидов
        for ast in env.asteroids:
            x, y = _project_to_screen_px(env, ast["pos"], width, height)
            r = _asteroid_screen_radius(env, ast["pos"])
            # Темнее, если дальше
            color_intensity = int(255 * (1.0 - np.clip(ast["pos"][1]/env.spawn_y, 0, 0.8)))
            pygame.draw.circle(screen, (color_intensity, color_intensity, color_intensity), (x, y), r)

        # Отрисовка пуль
        for p in env.projectiles:
            x, y = _project_to_screen_px(env, p["pos"], width, height)
            pygame.draw.circle(screen, (255, 255, 60), (x, y), 5)

        _draw_gun(screen, env, width, height)

        # Инфо текст
        text = font.render(f"Step Reward: {reward:+.2f} | Total: {total_reward:+.2f}", True, (255, 200, 100))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(30) # 30 FPS

        if done:
            obs, _ = env.reset()
            total_reward = 0.0

    pygame.quit()

if __name__ == "__main__":
    manual_mode()