import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.envs.registration import register

# Import your environment to register it
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.geometry_dash_env 

register(
    id='GeometryDash-v0',
    entry_point='src.geometry_dash_env:GeometryDashEnv', 
    max_episode_steps=1000,
)

def train_behavioral_cloning():
    data_path = "data/expert_data.npz"
    model_save_path = "models/gd_ppo_imitation"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path}. Did you run state_data.py?")

    # 1. Load the Expert Data
    print("Loading expert data...")
    data = np.load(data_path)
    states = torch.FloatTensor(data['states'])
    actions = torch.LongTensor(data['actions'])

    # Create a PyTorch DataLoader for batching
    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Initialize a fresh PPO Agent
    print("Initializing a fresh PPO agent...")
    env = gym.make('GeometryDash-v0')
    model = PPO("MlpPolicy", env, verbose=0)

    # 3. Hijack the PPO Policy for Supervised Learning
    print("\n" + "="*50)
    print("STARTING BEHAVIORAL CLONING (Brain Transfer)")
    print("="*50)
    
    # We use Adam optimizer directly on the PPO policy's neural network weights
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    epochs =35

    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_states, batch_actions in dataloader:
            # Ask the PPO policy what it would do given the expert's state
            distribution = model.policy.get_distribution(batch_states)
            
            # Calculate the loss: We want to maximize the probability of the expert's action
            log_prob = distribution.log_prob(batch_actions)
            loss = -log_prob.mean() 
            
            # Backpropagation (updating the weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Tracking accuracy for the logs
            total_loss += loss.item()
            predictions = distribution.get_actions(deterministic=True)
            correct_predictions += (predictions == batch_actions).sum().item()
            total_samples += len(batch_actions)
            
        accuracy = (correct_predictions / total_samples) * 100
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {accuracy:.1f}%")

    # 4. Save the pre-trained PPO Model
    os.makedirs("models", exist_ok=True)
    model.save(model_save_path)
    print("\n" + "="*50)
    print(f"SUCCESS! Cloned PPO agent saved to {model_save_path}.zip")
    print("="*50)

if __name__ == "__main__":
    train_behavioral_cloning()