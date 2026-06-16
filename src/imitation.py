import os
import cv2
import numpy as np
from ultralytics import YOLO
import sys
from tqdm import tqdm

# Add the project root to the path so we can import from perception
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extractor import FeatureExtractor

def process_expert_data():
    # 1. Configuration Paths
    frames_dir = "data/expert_frames"
    action_data_path = "data/imitation_data.npz"
    output_path = "data/expert_data.npz"
    yolo_model_path = "runs/detect/outputs/runs/geometry_dash_detector_v3/weights/best.pt" # Update if your weights are elsewhere

    # 2. Check if files exist
    if not os.path.exists(action_data_path):
        raise FileNotFoundError(f"Could not find {action_data_path}. Did you record actions?")
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"Could not find YOLO weights at {yolo_model_path}.")

    # 3. Load the raw actions and timestamps
    print(f"Loading raw action data from {action_data_path}...")
    data = np.load(action_data_path)
    actions = data['actions']
    
    total_frames = len(actions)
    print(f"Found {total_frames} recorded actions. Initializing YOLO...")

    # 4. Initialize AI models
    model = YOLO(yolo_model_path)
    extractor = FeatureExtractor(screen_width=1295, screen_height=810)

    states = []
    valid_actions = []
    
    print("\n" + "="*50)
    print("STARTING OFFLINE YOLO PROCESSING")
    print("="*50)

    # 5. Process each frame
    for i in tqdm(range(total_frames)):
        frame_path = os.path.join(frames_dir, f"frame_{i}.jpg")
        
        if not os.path.exists(frame_path):
            print(f"Warning: Missing {frame_path}. Skipping...")
            continue
            
        # Read the image
        img = cv2.imread(frame_path)
        
        # Run YOLO inference
        # verbose=False stops it from printing spam to your terminal for every single frame
        results = model.predict(img, verbose=False)

        _, _, normalized_features = extractor.extract(results)
        
        # Extract the 84-dim state vector
        state_vector = normalized_features["state_vector"]
        
        # Save the valid state and its corresponding action
        states.append(state_vector)
        valid_actions.append(actions[i])
        
    # 6. Save the final dataset
    print("\n" + "="*50)
    print("SAVING FINAL DATASET")
    print("="*50)
    
    states_np = np.array(states, dtype=np.float32)
    actions_np = np.array(valid_actions, dtype=np.int8)
    
    print(f"Final States shape: {states_np.shape} (Should be N, 84)")
    print(f"Final Actions shape: {actions_np.shape} (Should be N,)")
    
    np.savez(output_path, states=states_np, actions=actions_np)
    print(f"Dataset successfully saved to {output_path}! You are ready to train.")

if __name__ == "__main__":
    process_expert_data()