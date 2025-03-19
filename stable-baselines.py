import gym
from stable_baselines3 import PPO

# Create the environment
env = gym.make('CartPole-v1')

# Instantiate the PPO agent using a multilayer perceptron policy
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Reset the environment and unpack the observation and info
obs, _ = env.reset()

done = False
while not done:
    # Pass only the observation (not the tuple) to predict
    action, _states = model.predict(obs)
    # Step the environment; note the unpacking for Gymnasium's API
    obs, reward, terminated, truncated, info = env.step(action)
    # Combine the termination flags
    done = terminated or truncated
    env.render()

env.close()
