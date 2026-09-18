import os
import glob
import numpy as np
import SimpleITK as sitk

class ProstateXDataset:
    def __init__(self, data_dir, metadata_path=None):
        """
        Initialize ProstateX dataset loader for flat directory structure.

        Args:
            data_dir (str): Directory containing ProstateX-XXXX.nii and ProstateXMask-XXXX.nii files.
            metadata_path (str, optional): Ignored. Kept for backwards compatibility in init signature.
        """
        self.data_dir = data_dir

        # Discover all patient IDs based on image files
        search_pattern = os.path.join(self.data_dir, "ProstateX-*.nii*")
        img_files = glob.glob(search_pattern)

        self.patient_ids = []
        for f in img_files:
            filename = os.path.basename(f)
            # Extract XXXX from ProstateX-XXXX.nii or ProstateX-XXXX.nii.gz
            pid = filename.split('.')[0].replace("ProstateX-", "")
            self.patient_ids.append(pid)

        self.patient_ids = sorted(list(set(self.patient_ids)))

    def get_patient_data(self, patient_id):
        """
        Loads the single MRI volume and its corresponding mask.
        Aligns mask to MRI, normalizes MRI, and extracts target region centroid.
        """
        img_path = self._find_file(f"ProstateX-{patient_id}.nii")
        mask_path = self._find_file(f"ProstateXMask-{patient_id}.nii")

        if not img_path:
            raise FileNotFoundError(f"Missing MRI for patient {patient_id}: expected ProstateX-{patient_id}.nii")
        if not mask_path:
            raise FileNotFoundError(f"Missing mask for patient {patient_id}: expected ProstateXMask-{patient_id}.nii")

        mri_img = sitk.ReadImage(img_path)
        mask_img = sitk.ReadImage(mask_path)

        # Resample mask into MRI's physical space using nearest neighbor (to preserve categorical mask values)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(mri_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        mask_aligned = resampler.Execute(mask_img)

        # Verify alignment
        if mri_img.GetSize() != mask_aligned.GetSize():
            raise ValueError(f"Alignment failed for {patient_id}: sizes differ.")

        # Normalize MRI
        mri_norm = self._normalize(mri_img)

        # Convert to numpy arrays (Z, Y, X)
        mri_np = sitk.GetArrayFromImage(mri_norm)
        mask_np = sitk.GetArrayFromImage(mask_aligned)

        # Create single-channel volume: (1, Z, Y, X)
        volume = np.expand_dims(mri_np, axis=0)

        # Extract target region (prostate/lesion centroid) from mask
        # Note: We find the centroid of all non-zero voxels in the mask.
        targets = []
        non_zero_coords = np.argwhere(mask_np > 0)

        if len(non_zero_coords) > 0:
            centroid_z = int(np.mean(non_zero_coords[:, 0]))
            centroid_y = int(np.mean(non_zero_coords[:, 1]))
            centroid_x = int(np.mean(non_zero_coords[:, 2]))

            # Map to agent_pos convention [z, x, y]
            # SimpleITK numpy array is (z, y, x). Our env expects (z, x, y) conceptually,
            # where the last two dims are spatial W, H. Let's trace gymhisto:
            # env.agent_pos = [z, x, y].
            targets.append({
                'coord': [centroid_z, centroid_x, centroid_y],
                'score': 3 # Default clinical score representing generic target
            })

        return volume, targets, mri_img

    def _find_file(self, filename_prefix):
        """Finds a file matching the prefix (handling .nii or .nii.gz)."""
        pattern = os.path.join(self.data_dir, f"{filename_prefix}*")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        return None

    def _normalize(self, image):
        """Normalize image intensities using Z-score."""
        img_float = sitk.Cast(image, sitk.sitkFloat32)
        stats = sitk.StatisticsImageFilter()
        stats.Execute(img_float)
        mean, std = stats.GetMean(), stats.GetSigma()

        if std == 0:
            return img_float

        normalized = sitk.ShiftScale(img_float, shift=-mean, scale=1.0/std)
        return normalized
