import gym
import numpy as np
import random
from gym import spaces
from stable_baselines3 import PPO

# Define a simple Tic Tac Toe environment
class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        # The board is a 3x3 grid flattened to a 9-element array.
        # Empty cells = 0, agent's moves = 1 (X), opponent's moves = -1 (O)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)
        # Nine possible moves (each cell)
        self.action_space = spaces.Discrete(9)
        self.reset()
        
    def reset(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # Agent is player 1
        return self.board.copy()
    
    def step(self, action):
        # Check for valid move
        if self.board[action] != 0:
            # Invalid move gives a penalty and ends the episode
            return self.board.copy(), -10, True, {"invalid": True}
        
        # Agent makes a move
        self.board[action] = self.current_player
        if self.check_winner(self.current_player):
            return self.board.copy(), 1, True, {}
        if np.all(self.board != 0):
            return self.board.copy(), 0, True, {}  # Draw
        
        # Opponent's turn: choose a random available move
        available_moves = [i for i in range(9) if self.board[i] == 0]
        opponent_action = random.choice(available_moves)
        self.board[opponent_action] = -self.current_player
        if self.check_winner(-self.current_player):
            return self.board.copy(), -1, True, {}
        if np.all(self.board != 0):
            return self.board.copy(), 0, True, {}
        
        return self.board.copy(), 0, False, {}
    
    def check_winner(self, player):
        b = self.board.reshape(3, 3)
        # Check rows and columns
        for i in range(3):
            if np.all(b[i, :] == player) or np.all(b[:, i] == player):
                return True
        # Check diagonals
        if b[0, 0] == player and b[1, 1] == player and b[2, 2] == player:
            return True
        if b[0, 2] == player and b[1, 1] == player and b[2, 0] == player:
            return True
        return False
    
    def render(self, mode='human'):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        board = self.board.reshape(3, 3)
        for row in board:
            print("|".join([symbols[cell] for cell in row]))
            print("-----")

# Function to perform Monte Carlo rollouts using the agent's policy
def monte_carlo_rollout(agent, env, num_rollouts=100):
    rewards = []
    for _ in range(num_rollouts):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Use the policy to pick an action (deterministic for evaluation)
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

# Create the Tic Tac Toe environment
env = TicTacToeEnv()

# Initialize the PPO agent with a Multi-Layer Perceptron policy
agent = PPO("MlpPolicy", env, verbose=0)

# Train the agent for demonstration (adjust timesteps as needed)
agent.learn(total_timesteps=10000)

# Run Monte Carlo rollouts to evaluate the agent's performance
avg_reward = monte_carlo_rollout(agent, env, num_rollouts=100)
print("Average reward over 100 rollouts:", avg_reward)
