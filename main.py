from env import AsteroidDefenseEnv
from visual_ursina import UrsinaRenderer

if __name__ == "__main__":
    env = AsteroidDefenseEnv(config={})
    renderer = UrsinaRenderer(width=1024, height=768)
    
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    
    while not done:
        # Для теста: случайные действия пушки
        action = env.action_space.sample() 
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        # Если нажали ESC или закрыли окно - останавливаем
        if not renderer.process_events():
            break
            
        renderer.draw(env, reward=reward, total_reward=total_reward, fps=60)
        
    renderer.close()