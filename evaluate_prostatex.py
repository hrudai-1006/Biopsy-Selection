import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from gymhisto import HistoEnv

# ---------------------------------------------------------
# Evaluation Setup
# ---------------------------------------------------------
# Set this to your Kaggle data directory
DATA_DIR = "/kaggle/input/datasets/veraoliveira/prostatex"
MODEL_PATH = "ppo_prostatex_model.zip"

def get_kaggle_patients(data_dir, num_patients=20):
    """Finds available patients in the Kaggle directory."""
    search_pattern = os.path.join(data_dir, "ProstateX-*.nii*")
    img_files = glob.glob(search_pattern)
    patient_ids = [os.path.basename(f).split('.')[0].replace("ProstateX-", "") for f in img_files]
    return sorted(list(set(patient_ids)))[:num_patients]

def evaluate_agent(env, model=None, is_random=False, max_steps=1000):
    """Runs a single episode and returns metrics."""
    obs, info = env.reset()

    # Store initial target position from environment for distance calculation
    if not env.targets:
        return None # Skip if no targets

    # Get the closest target to center
    target_pos = env.targets[0]['coord']

    steps = 0
    final_overlap = 0.0

    while steps < max_steps:
        if is_random:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        step_res = env.step(action)

        # Handle API differences
        if len(step_res) == 5:
            obs, reward, done, truncated, info = step_res
        else:
            obs, reward, done, info = step_res
            truncated = False

        final_overlap, _ = env._calculate_prostatex_scores(env.agent_pos)
        steps += 1

        if done or truncated:
            break

    # Calculate final distance (Euclidean distance between final agent_pos and target_pos)
    # Both are [z, x, y]
    final_pos = env.agent_pos
    dist = np.sqrt((final_pos[0] - target_pos[0])**2 +
                   (final_pos[1] - target_pos[1])**2 +
                   (final_pos[2] - target_pos[2])**2)

    # Hit is defined as being within the threshold used in the reward function (e.g. 20 voxels)
    is_hit = dist < 20.0

    return {
        "distance": dist,
        "overlap": final_overlap,
        "hit": is_hit,
        "steps": steps
    }

def run_evaluation():
    patients = get_kaggle_patients(DATA_DIR)
    if not patients:
        print(f"Error: No patients found in {DATA_DIR}. Please check the path.")
        return

    print(f"Starting evaluation on {len(patients)} patients...")

    # Initialize environment
    env = HistoEnv(
        img_path="dummy", xml_path="dummy", tile_size=64, result_path="./tmp",
        mode="prostatex", prostatex_data_dir=DATA_DIR
    )

    # Load Model
    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH)
        print("Trained model loaded successfully.")
    else:
        print("Warning: Trained model not found. Evaluation will fail for PPO agent.")
        return

    results = []

    for pid in patients:
        print(f"Evaluating Patient {pid}...")
        env.patient_id = pid
        try:
            env._load_patient(pid)

            # PPO Agent Evaluation
            ppo_metrics = evaluate_agent(env, model=model, is_random=False)
            if ppo_metrics:
                ppo_metrics['agent'] = 'PPO Trained'
                ppo_metrics['patient'] = pid
                results.append(ppo_metrics)

            # Random Baseline Evaluation
            env._load_patient(pid) # Reload to reset safely
            rnd_metrics = evaluate_agent(env, is_random=True)
            if rnd_metrics:
                rnd_metrics['agent'] = 'Random Baseline'
                rnd_metrics['patient'] = pid
                results.append(rnd_metrics)

        except Exception as e:
            print(f"Skipping {pid} due to error: {e}")

    df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Display Text Metrics
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS")
    print("="*50)

    summary = df.groupby('agent').agg(
        Hit_Rate=('hit', 'mean'),
        Mean_Distance=('distance', 'mean'),
        Mean_Overlap=('overlap', 'mean'),
        Mean_Steps=('steps', 'mean')
    ).round(3)

    summary['Hit_Rate'] = summary['Hit_Rate'].apply(lambda x: f"{x*100:.1f}%")
    print(summary.to_string())
    print("="*50)

    # ---------------------------------------------------------
    # Plot Graphs
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PPO Agent vs Random Baseline Performance', fontsize=18, y=1.02)

    # 1. Hit Rate Bar Plot
    hit_rate = df.groupby('agent')['hit'].mean() * 100
    sns.barplot(x=hit_rate.index, y=hit_rate.values, ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title('Malignancy Hit Rate (%)', fontsize=14)
    axes[0, 0].set_ylabel('Hit Rate (%)')
    axes[0, 0].set_ylim(0, 100)

    # 2. Distance Box Plot
    sns.boxplot(data=df, x='agent', y='distance', ax=axes[0, 1], palette="viridis")
    axes[0, 1].set_title('Distance to Target (Lower is better)', fontsize=14)
    axes[0, 1].set_ylabel('Voxels')

    # 3. Overlap Score Violin Plot
    sns.violinplot(data=df, x='agent', y='overlap', ax=axes[1, 0], palette="viridis")
    axes[1, 0].set_title('Geometric Overlap Score (Higher is better)', fontsize=14)
    axes[1, 0].set_ylabel('Score (0 to 1)')

    # 4. Steps to Converge Histogram
    sns.histplot(data=df, x='steps', hue='agent', element="step", stat="density", common_norm=False, ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title('Steps to Converge/Terminate', fontsize=14)
    axes[1, 1].set_xlabel('Steps')

    plt.tight_layout()
    plt.savefig('evaluation_metrics_graphs.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nGraphs saved to 'evaluation_metrics_graphs.png'.")

if __name__ == "__main__":
    # If running in a notebook, ensure paths are correct before executing.
    run_evaluation()
