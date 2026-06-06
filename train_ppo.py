import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymhisto import HistoEnv

def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent on ProstateX")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to ProstateX data directory")
    parser.add_argument('--metadata', type=str, default=None, help="Path to ProstateX metadata CSV")
    parser.add_argument('--timesteps', type=int, default=10000, help="Total timesteps to train")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Directory {args.data_dir} does not exist. Please provide valid path.")
        return

    env = HistoEnv(
        img_path="dummy.tif",
        xml_path="dummy.xml",
        tile_size=64,
        result_path="./prostatex_results",
        mode="prostatex",
        prostatex_data_dir=args.data_dir,
        prostatex_metadata=args.metadata
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

    model.save("ppo_prostatex_model")
    print("Model saved to ppo_prostatex_model.zip")

    print("Running evaluation episode...")
    obs = env.reset()
    for _ in range(50):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render(mode="save")
        if done:
            break

if __name__ == "__main__":
    main()
