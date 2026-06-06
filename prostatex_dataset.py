import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

class ProstateXDataset:
    def __init__(self, data_dir, metadata_path=None):
        """
        Initialize ProstateX dataset loader.

        Args:
            data_dir (str): Directory containing MRI patient folders.
            metadata_path (str, optional): Path to metadata CSV with lesion info.
        """
        self.data_dir = data_dir
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)

    def get_patient_data(self, patient_id):
        """
        Load T2W, DWI, and DCE volumes for a patient.
        Aligns them to a common physical space (e.g., T2W's space).
        """
        patient_dir = os.path.join(self.data_dir, patient_id)
        if not os.path.exists(patient_dir):
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        t2w_path = self._find_modality(patient_dir, 't2w')
        dwi_path = self._find_modality(patient_dir, 'dwi')
        dce_path = self._find_modality(patient_dir, 'dce')

        t2w_img = sitk.ReadImage(t2w_path)
        dwi_img = sitk.ReadImage(dwi_path)
        dce_img = sitk.ReadImage(dce_path)

        # Resample and align DWI and DCE to T2W
        dwi_aligned = self._resample_to_reference(dwi_img, t2w_img)
        dce_aligned = self._resample_to_reference(dce_img, t2w_img)

        # Normalize
        t2w_norm = self._normalize(t2w_img)
        dwi_norm = self._normalize(dwi_aligned)
        dce_norm = self._normalize(dce_aligned)

        # Extract numpy arrays (Z, Y, X)
        t2w_np = sitk.GetArrayFromImage(t2w_norm)
        dwi_np = sitk.GetArrayFromImage(dwi_norm)
        dce_np = sitk.GetArrayFromImage(dce_norm)

        # Stack to (3, Z, Y, X)
        stacked = np.stack([t2w_np, dwi_np, dce_np], axis=0)

        # Get lesions if available
        lesions = []
        if self.metadata is not None:
            patient_lesions = self.metadata[self.metadata['ProxID'] == patient_id]
            for _, row in patient_lesions.iterrows():
                # ProstateX metadata gives physical points. Convert to index
                # assuming pos_x, pos_y, pos_z are physical coordinates.
                physical_point = (row['pos_x'], row['pos_y'], row['pos_z'])
                index_point = t2w_img.TransformPhysicalPointToIndex(physical_point)

                # SimpleITK returns index as (x, y, z), we need to ensure it maps to
                # our environment agent_pos which is [z, x, y]
                idx_x, idx_y, idx_z = index_point

                score = row.get('ClinSig', row.get('PIRADS', 3)) # Default to moderate if not found
                lesions.append({'coord': [idx_z, idx_x, idx_y], 'score': score})

        return stacked, lesions, t2w_img # Returning image for metadata/spacing info

    def _find_modality(self, patient_dir, modality):
        """Helper to find the file for a specific modality."""
        for root, dirs, files in os.walk(patient_dir):
            for file in files:
                if modality.lower() in file.lower() and file.endswith(('.nii.gz', '.mha', '.nrrd', '.dcm')):
                    return os.path.join(root, file)
        return os.path.join(patient_dir, f"{modality}.nii.gz")

    def _resample_to_reference(self, image, reference):
        """Resample an image to match the physical space of a reference image."""
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        return resampler.Execute(image)

    def _normalize(self, image):
        """Normalize image intensities (e.g., Z-score or min-max)."""
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
