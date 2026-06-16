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
    # Load your newly cloned brain!
    model = PPO.load("models/gd_ppo_imitation.zip")
    
    obs, info = env.reset()
    
    print("Agent is taking the wheel! Press Ctrl+C to stop.")
    try:
        while True:
            # Let the agent predict the action
            action, _states = model.predict(obs, deterministic=True)
            print(f"Agent wants to: {'JUMP' if action == 1 else 'Wait'} | Distance to spike: {obs[4]:.2f}") # obs[4] is usually the next spike distance
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("Agent died. Restarting...")
                time.sleep(1) # Pause so you can see what killed it
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("Stopping evaluation.")
    finally:
        env.close()

if __name__ == "__main__":
    play()