import os
import argparse
import numpy as np
import SimpleITK as sitk
from stable_baselines3 import PPO
from gymhisto import HistoEnv
import matplotlib.pyplot as plt
import cv2
import shutil

def load_and_preprocess_nii(nii_path):
    """Loads a single .nii file and normalizes it."""
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

    # New loader expects (1, Z, Y, X)
    stacked = np.expand_dims(np_img, axis=0)
    return stacked, sitk.GetArrayFromImage(img)

class DummyProstateXDataset:
    def __init__(self, volume_mri, patient_id="MockPatient"):
        self.volume_mri = volume_mri
        self.patient_ids = [patient_id]

    def get_patient_data(self, patient_id):
        # We just return the pre-loaded, stacked MRI, empty targets, and None for img
        return self.volume_mri, [], None

def annotate_biopsy_position(slice_image, position, box_size=64, label="Predicted Biopsy Position"):
    """
    Programmatically draws the RL agent's selected position on the MRI slice using OpenCV.

    Args:
        slice_image (np.ndarray): 2D array of the MRI slice.
        position (tuple): (x, y) coordinates of the agent.
        box_size (int): Size of the target bounding box.
        label (str): Text label for the box.

    Returns:
        np.ndarray: BGR annotated image.
    """
    x, y = int(position[0]), int(position[1])
    h, w = slice_image.shape

    # Normalize MRI slice for display (0-255 uint8)
    norm_slice = cv2.normalize(slice_image, None, 0, 255, cv2.NORM_MINMAX)
    uint8_slice = norm_slice.astype(np.uint8)

    # Convert grayscale to BGR for colored annotations
    annotated = cv2.cvtColor(uint8_slice, cv2.COLOR_GRAY2BGR)

    # Determine bounding box coordinates, clamped to image boundaries
    half_box = box_size // 2
    x1 = max(0, x - half_box)
    y1 = max(0, y - half_box)
    x2 = min(w - 1, x + half_box)
    y2 = min(h - 1, y + half_box)

    # Validation constraint check
    if not (0 <= x < w and 0 <= y < h):
        print(f"Warning: Selected position (x={x}, y={y}) is outside image boundaries (w={w}, h={h})")

    # Draw red bounding box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw a small center dot and crosshair
    cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)
    crosshair_len = 5
    cv2.line(annotated, (x - crosshair_len, y), (x + crosshair_len, y), (0, 255, 0), 1)
    cv2.line(annotated, (x, y - crosshair_len), (x, y + crosshair_len), (0, 255, 0), 1)

    # Add text label above the box
    text_y = max(10, y1 - 5)
    cv2.putText(annotated, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return annotated

def generate_attention_overlay(slice_image, attention_map, position, box_size=64):
    """
    Generates an attention heatmap overlaid onto the original MRI slice.
    """
    x, y = int(position[0]), int(position[1])
    h, w = slice_image.shape

    # Normalize MRI slice for display (0-255 uint8)
    norm_slice = cv2.normalize(slice_image, None, 0, 255, cv2.NORM_MINMAX)
    uint8_slice = norm_slice.astype(np.uint8)
    rgb_slice = cv2.cvtColor(uint8_slice, cv2.COLOR_GRAY2RGB)

    # Base heatmap mask
    heatmap_mask = np.zeros((h, w), dtype=np.float32)

    if attention_map is not None:
        attention_resized = cv2.resize(attention_map, (box_size, box_size))

        half_box = box_size // 2
        x1 = max(0, x - half_box)
        y1 = max(0, y - half_box)
        x2 = min(w, x + half_box)
        y2 = min(h, y + half_box)

        patch_w = x2 - x1
        patch_h = y2 - y1

        if patch_w > 0 and patch_h > 0:
            att_x_start = max(0, half_box - x)
            att_y_start = max(0, half_box - y)
            heatmap_mask[y1:y2, x1:x2] = attention_resized[att_y_start:att_y_start+patch_h, att_x_start:att_x_start+patch_w]

    # Apply jet colormap
    heatmap_colored = cv2.applyColorMap((heatmap_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Create an alpha mask to only blend where heatmap intensity > threshold
    alpha = np.where(heatmap_mask > 0.1, 0.4, 0.0)[..., np.newaxis]

    # Blend overlay with original slice
    blended = (rgb_slice * (1.0 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return blended

def main():
    parser = argparse.ArgumentParser(description="Predict Prostate Target using PPO Agent")
    parser.add_argument('--input_nii', type=str, required=True, help="Path to input MRI .nii or .nii.gz file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained PPO model (.zip)")
    parser.add_argument('--output_dir', type=str, default="./outputs", help="Directory to save the prediction image")
    parser.add_argument('--max_steps', type=int, default=500, help="Maximum number of steps for inference")
    parser.add_argument('--patient_id', type=str, default="0000", help="Patient ID for output naming")
    args = parser.parse_args()

    # Structured output directory
    patient_output_dir = os.path.join(args.output_dir, "prostatex", f"patient_{args.patient_id}")
    os.makedirs(patient_output_dir, exist_ok=True)

    print(f"Loading MRI scan from {args.input_nii}...")
    volume_mri, original_mri = load_and_preprocess_nii(args.input_nii)

    # Create an empty mock directory to pass initialization check
    mock_dir = "temp_mock_prostatex"
    patient_id = args.patient_id
    os.makedirs(mock_dir, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(volume_mri[0]), os.path.join(mock_dir, f"ProstateX-{patient_id}.nii"))
    sitk.WriteImage(sitk.GetImageFromArray(volume_mri[0]), os.path.join(mock_dir, f"ProstateXMask-{patient_id}.nii"))

    try:
        env = HistoEnv(
            img_path="dummy", xml_path="dummy", tile_size=64, result_path=patient_output_dir,
            mode="prostatex", prostatex_data_dir=mock_dir
        )

        # Override the dataset with our dummy dataset containing the raw MRI
        env.dataset = DummyProstateXDataset(volume_mri, patient_id)

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
        z, x, y = int(final_pos[0]), int(final_pos[1]), int(final_pos[2])
        print(f"Patient: {patient_id}")
        print(f"Selected 3D position: [z={z}, y={y}, x={x}]")
        print(f"Displayed slice: z={z}")
        print(f"2D display position: (x={x}, y={y})")

        # Extract corresponding 2D slice
        slice_img = original_mri[z]
        h, w = slice_img.shape
        print(f"Image dimensions: (width={w}, height={h})")

        # Determine actual bounding box
        half_box = env.tile_size // 2
        print(f"Box: (x1={max(0, x - half_box)}, y1={max(0, y - half_box)}, x2={min(w-1, x + half_box)}, y2={min(h-1, y + half_box)})")

        attention_map = env.current_attention_map

        # Generate Annotated Image (OpenCV)
        annotated_img_bgr = annotate_biopsy_position(slice_img, (x, y), env.tile_size)
        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)

        # Generate Attention Overlay (OpenCV)
        attention_overlay_rgb = generate_attention_overlay(slice_img, attention_map, (x, y), env.tile_size)

        # Save individual outputs
        base_filename = f"patient_{patient_id}_z{z}_x{x}_y{y}"
        cv2.imwrite(os.path.join(patient_output_dir, f"{base_filename}_annotated.png"), annotated_img_bgr)
        cv2.imwrite(os.path.join(patient_output_dir, f"{base_filename}_attention.png"), cv2.cvtColor(attention_overlay_rgb, cv2.COLOR_RGB2BGR))

        # Generate Combined Visualization Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f"Patient {patient_id} — Predicted Biopsy Position\nPosition: [z={z}, y={y}, x={x}]", fontsize=16)

        ax1.imshow(annotated_img_rgb)
        ax1.set_title("MRI Slice")
        ax1.axis('off')

        ax2.imshow(attention_overlay_rgb)
        ax2.set_title("Attention Heatmap")
        ax2.axis('off')

        combined_out_path = os.path.join(patient_output_dir, f"{base_filename}_combined.png")
        plt.tight_layout()
        plt.savefig(combined_out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print(f"Saved combined visualization to {combined_out_path}")

    finally:
        shutil.rmtree(mock_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
