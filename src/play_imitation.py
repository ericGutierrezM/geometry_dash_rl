import os
import time
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.envs.registration import register
import geometry_dash_env 

# Register the environment
register(
    id='GeometryDash-v0',
    entry_point='geometry_dash_env:GeometryDashEnv',
    max_episode_steps=1000,
)

def play():
    print("Loading environment...")
    env = gym.make('GeometryDash-v0')
    
    print("Loading Imitation brain...")
    model = PPO.load("ppo_gd_simulado_experto.zip")
    
    obs, info = env.reset()
    
    # --- MEMORY INITIALIZATION (Track both separately!) ---
    last_known_spike = 1.0
    last_known_block = 1.0
    
    print("Agent is taking the wheel! Press Ctrl+C to stop.")
    try:
        while True:
            # --- 1. SPIKE PERMANENCE (obs[4]) ---
            if obs[4] >= 0.99 and last_known_spike < 1.0:
                guessed_spike = max(0.0, last_known_spike - 0.15)
                obs[4] = guessed_spike 
                if guessed_spike == 0.0:
                    last_known_spike = 1.0
            
            if obs[4] > 0.0: 
                last_known_spike = obs[4]

            # --- 2. BLOCK PERMANENCE (obs[5]) ---
            if obs[5] >= 0.99 and last_known_block < 1.0:
                guessed_block = max(0.0, last_known_block - 0.15)
                obs[5] = guessed_block 
                if guessed_block == 0.0:
                    last_known_block = 1.0
            
            if obs[5] > 0.0: 
                last_known_block = obs[5]

            # --- 3. LET THE AGENT DECIDE ---
            action, _states = model.predict(obs, deterministic=True)

            # --- 4. UNIVERSAL PEACE TIME OVERRIDE ---
            # Find the absolute closest obstacle (Spike OR Block)
            nearest_obstacle = min(obs[4], obs[5])
            
            # If BOTH are completely off-screen, force wait.
            if nearest_obstacle >= 0.99:
                action = 0

            # Debugging print to see what the AI is reacting to
            target = "Spike" if obs[4] < obs[5] else "Block"
            dist = nearest_obstacle
            print(f"Agent wants to: {'JUMP' if action == 1 else 'Wait'} | Nearest {target}: {dist:.2f}") 
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("Agent died. Restarting...")
                time.sleep(1) 
                obs, info = env.reset()
                
                # Wipe both memories on death
                last_known_spike = 1.0 
                last_known_block = 1.0 
                
    except KeyboardInterrupt:
        print("Stopping evaluation.")
    finally:
        env.close()

if __name__ == "__main__":
    play()