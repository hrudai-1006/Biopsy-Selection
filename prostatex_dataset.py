import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass

class ProstateXDataset:
    def __init__(self, data_dir, metadata_path=None):
        """
        Initialize ProstateX dataset loader.
        The dataset is a flat folder with ProstateX-XXXX.nii and ProstateXMask-XXXX.nii files.

        Args:
            data_dir (str): Directory containing MRI .nii files.
            metadata_path (str, optional): Not needed anymore.
        """
        self.data_dir = data_dir

    def get_patient_data(self, patient_id):
        """
        Load MRI volume for a patient.
        Stacks the single MRI 3x to simulate T2W/DWI/DCE channels giving shape (3, Z, Y, X).
        Extracts lesion coordinates from mask centroid.
        """
        mri_path = os.path.join(self.data_dir, f"{patient_id}.nii")
        mask_path = os.path.join(self.data_dir, patient_id.replace("ProstateX", "ProstateXMask") + ".nii")

        if not os.path.exists(mri_path):
            raise FileNotFoundError(f"MRI file not found: {mri_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        mri_img = sitk.ReadImage(mri_path)
        mask_img = sitk.ReadImage(mask_path)

        # Normalize MRI
        mri_norm = self._normalize(mri_img)

        # Extract numpy array (Z, Y, X)
        mri_np = sitk.GetArrayFromImage(mri_norm)
        mask_np = sitk.GetArrayFromImage(mask_img)

        # Stack to (3, Z, Y, X)
        stacked = np.stack([mri_np, mri_np, mri_np], axis=0)

        # Get lesions from mask
        lesions = []
        if np.any(mask_np):
            # Calculate centroid of the binary mask
            # mask_np is of shape (Z, Y, X)
            z, y, x = center_of_mass(mask_np > 0)
            lesions.append({'coord': [int(round(z)), int(round(x)), int(round(y))], 'score': 3}) # default score

        return stacked, lesions, mri_img

    def _normalize(self, image):
        """Normalize image intensities (e.g., Z-score)."""
        img_float = sitk.Cast(image, sitk.sitkFloat32)
        mean, std = self._get_image_stats(img_float)
        if std == 0:
            return img_float
        normalized = sitk.ShiftScale(img_float, shift=-mean, scale=1.0/std)
        return normalized

    def _get_image_stats(self, image):
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        return stats.GetMean(), stats.GetSigma()
