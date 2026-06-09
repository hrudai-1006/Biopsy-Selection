import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymhisto import HistoEnv

def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent")
    parser.add_argument('--mode', type=str, default="prostatex", help="Mode: prostatex or saras")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to ProstateX data directory")
    parser.add_argument('--saras_data_dir', type=str, default=None, help="Path to SARAS data directory")
    parser.add_argument('--metadata', type=str, default=None, help="Path to ProstateX metadata CSV")
    parser.add_argument('--timesteps', type=int, default=100000, help="Total timesteps to train")
    args = parser.parse_args()

    if args.mode == "prostatex" and not args.data_dir:
        print("Error: --data_dir is required for prostatex mode.")
        return
    if args.mode == "saras" and not args.saras_data_dir:
        print("Error: --saras_data_dir is required for saras mode.")
        return

    result_dir = f"./{args.mode}_results"

    env = HistoEnv(
        img_path="dummy.tif",
        xml_path="dummy.xml",
        tile_size=64,
        result_path=result_dir,
        mode=args.mode,
        prostatex_data_dir=args.data_dir,
        prostatex_metadata=args.metadata,
        saras_data_dir=args.saras_data_dir
    )

    print("Checking environment...")
    try:
        check_env(env, warn=True)
        print("Environment check passed!")
    except Exception as e:
        print(f"Warning: Environment check failed: {e}")

    print("Initializing PPO model...")
    model = PPO("MlpPolicy", env, verbose=1)

    print(f"Training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    model_name = f"ppo_{args.mode}_model"
    model.save(model_name)
    print(f"Model saved to {model_name}.zip")

    print("Running evaluation episode...")
    obs, info = env.reset()

    hits = 0
    overlap_scores = []
    distances = []

    for _ in range(50):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render(mode="save")

        if args.mode == "saras" and 'overlap_score' in info:
            overlap_scores.append(info['overlap_score'])
            if info['overlap_score'] > 0.5:
                hits += 1
        elif args.mode == "prostatex" and 'overlap_score' in info:
            overlap_scores.append(info['overlap_score'])
            if info['overlap_score'] > 0.5:
                hits += 1

        # Calculate distance to center if we are in prostatex and position is returned
        if args.mode == "prostatex" and 'position' in info and hasattr(env, 'lesions'):
            z, x, y = info['position']
            min_dist = float('inf')
            for lesion in env.lesions:
                lz, lx, ly = lesion['coord']
                dist = np.sqrt((z - lz)**2 + (y - ly)**2 + (x - lx)**2)
                if dist < min_dist:
                    min_dist = dist
            if min_dist != float('inf'):
                distances.append(min_dist)

        if done or truncated:
            break

    if len(overlap_scores) > 0:
        print(f"Evaluation Metrics:")
        print(f"  Hit rate (overlap > 0.5): {hits/len(overlap_scores):.2f}")
        print(f"  Mean overlap score: {np.mean(overlap_scores):.2f}")

    if len(distances) > 0:
        print(f"  Mean distance: {np.mean(distances):.2f}")

if __name__ == "__main__":
    main()
