import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()