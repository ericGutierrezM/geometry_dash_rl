import gymnasium as gym
from gymnasium.envs.registration import register
import geometry_dash_env # This registers your environment
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

register(
    id='GeometryDash-v0',
    entry_point='geometry_dash_env:GeometryDashEnv', # Ensure this matches your filename:ClassName
    max_episode_steps=1000,
)                            

def train():
    # 1. Create the environment
    env = gym.make('GeometryDash-v0')
    model_path = "./models/gd_ppo_imitation.zip"

    # 2. Setup directory to save models
    log_dir = "./logs/"
    model_dir = "./models/"
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = PPO.load(model_path, 
                         env=env, 
                         verbose=1, 
                         learning_rate=0.0003,
                         batch_size=64,
                         gamma=0.99,
                         tensorboard_log=log_dir)
    else:
        print("No model found. Initializing a new agent...")
        model = PPO("MlpPolicy", 
                    env=env, 
                    verbose=1, 
                    learning_rate=0.0003,
                    batch_size=64,
                    gamma=0.99,
                    tensorboard_log=log_dir)

    # 3. Callback to save periodically (prevents losing progress if game crashes)
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=model_dir,
        name_prefix='gd_ppo_checkpoint'
    )


    print("Agent initialized. Starting training loop...")
    
    # 5. Training loop
    # We set a callback so that every 2000 steps, we save the model
    try:
        model.learn(
            total_timesteps=50000, 
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name="PPO_4"
        )
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current progress...")
    finally:
        model.save(f"{model_dir}/gd_ppo_final_w_imitation")
        env.close()
        print("Final model saved. Environment closed.")

if __name__ == "__main__":
    train()