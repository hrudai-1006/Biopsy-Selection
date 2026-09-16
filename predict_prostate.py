import os
import argparse
import numpy as np
import SimpleITK as sitk
from stable_baselines3 import PPO
from gymhisto import HistoEnv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import shutil

def load_and_preprocess_nii(nii_path):
    """Loads a single .nii file, normalizes it, and stacks it 3x to simulate modalities."""
    img = sitk.ReadImage(nii_path)
    img_float = sitk.Cast(img, sitk.sitkFloat32)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(img_float)
    mean, std = stats.GetMean(), stats.GetSigma()
    if std != 0:
        normalized = sitk.ShiftScale(img_float, shift=-mean, scale=1.0/std)
    else:
        normalized = img_float

    np_img = sitk.GetArrayFromImage(normalized)

    # Stack 3 times to simulate T2W, DWI, DCE
    stacked = np.stack([np_img, np_img, np_img], axis=0)
    return stacked, sitk.GetArrayFromImage(img)

class DummyProstateXDataset:
    def __init__(self, stacked_mri):
        self.stacked_mri = stacked_mri

    def get_patient_data(self, patient_id):
        # We just return the pre-loaded, stacked MRI, empty lesions, and None for img
        return self.stacked_mri, [], None


def main():
    parser = argparse.ArgumentParser(description="Predict Prostate Lesion using PPO Agent")
    parser.add_argument('--input_nii', type=str, required=True, help="Path to input MRI .nii or .nii.gz file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained PPO model (.zip)")
    parser.add_argument('--output_dir', type=str, default="./predictions", help="Directory to save the prediction image")
    parser.add_argument('--max_steps', type=int, default=500, help="Maximum number of steps for inference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading MRI scan from {args.input_nii}...")
    stacked_mri, original_mri = load_and_preprocess_nii(args.input_nii)

    # Create an empty mock directory to pass initialization check
    # HistoEnv will not use it because we override the dataset
    mock_dir = "temp_mock_prostatex"
    patient_id = "MockPatient"
    patient_dir = os.path.join(mock_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(stacked_mri[0]), os.path.join(patient_dir, "t2w.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(stacked_mri[1]), os.path.join(patient_dir, "dwi.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(stacked_mri[2]), os.path.join(patient_dir, "dce.nii.gz"))

    patient_id = "MockPatient"
    os.makedirs(os.path.join(mock_dir, patient_id), exist_ok=True)

    try:
        # Note: the older gym API vs gymnasium API can cause unpack errors if not handled correctly.
        # HistoEnv currently uses gymnasium (we patched it), but some old gym imports might remain in stable_baselines3.
        # It's safest to rely on the fact that HistoGym takes these parameters.
        env = HistoEnv(
            img_path="dummy", xml_path="dummy", tile_size=64, result_path=args.output_dir,
            mode="prostatex", prostatex_data_dir=mock_dir, prostatex_metadata=None
        )

        # Override the dataset with our dummy dataset containing the raw MRI
        env.dataset = DummyProstateXDataset(stacked_mri)

        # We need to manually call load patient since we bypass standard logic
        env.patient_id = patient_id
        env._load_patient(patient_id)

        # Reset returns obs, info in Gymnasium
        reset_res = env.reset()
        if isinstance(reset_res, tuple) and len(reset_res) == 2:
            obs = reset_res[0]
        else:
            obs = reset_res

        print(f"Loading model from {args.model_path}...")
        model = PPO.load(args.model_path)

        print("Starting inference...")
        for step in range(args.max_steps):
            action, _states = model.predict(obs, deterministic=True)
            step_res = env.step(action)

            # Handle both gym and gymnasium step return signatures
            if len(step_res) == 5:
                obs, reward, done, truncated, info = step_res
                if done or truncated: break
            elif len(step_res) == 4:
                obs, reward, done, info = step_res
                if done: break

        final_pos = env.agent_pos
        print(f"Agent converged at position (z, x, y): {final_pos}")

        # Visualization
        z, x, y = final_pos
        slice_img = original_mri[z]
        attention_map = env.current_attention_map

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(slice_img, cmap='gray')

        # Get dimensions from the volume itself
        max_z, max_y, max_x = env.volume.shape[1], env.volume.shape[2], env.volume.shape[3]

        # Heatmap overlay
        if attention_map is not None:
            # Resize attention map to patch size
            attention_resized = cv2.resize(attention_map, (env.tile_size, env.tile_size))
            # Place it at the agent's position
            heatmap_overlay = np.zeros_like(slice_img, dtype=np.float32)

            x_min = max(0, x - env.tile_size // 2)
            x_max = min(max_x, x + env.tile_size // 2)
            y_min = max(0, y - env.tile_size // 2)
            y_max = min(max_y, y + env.tile_size // 2)

            patch_w = x_max - x_min
            patch_h = y_max - y_min

            if patch_w > 0 and patch_h > 0:
                # Crop attention if agent is at the edge
                att_x_start = max(0, env.tile_size // 2 - x)
                att_y_start = max(0, env.tile_size // 2 - y)
                heatmap_overlay[y_min:y_max, x_min:x_max] = attention_resized[att_y_start:att_y_start+patch_h, att_x_start:att_x_start+patch_w]

                # Make the heatmap semi-transparent
                masked_data = np.ma.masked_where(heatmap_overlay < 0.1, heatmap_overlay)
                ax.imshow(masked_data, cmap='jet', alpha=0.4)

        # Bounding box
        rect = patches.Rectangle((x - env.tile_size//2, y - env.tile_size//2), env.tile_size, env.tile_size, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"Predicted Biopsy Site (Slice {z})")
        ax.axis('off')

        out_path = os.path.join(args.output_dir, f"prediction_slice_{z}.png")
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Saved prediction visualization to {out_path}")

    finally:
        shutil.rmtree(mock_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
