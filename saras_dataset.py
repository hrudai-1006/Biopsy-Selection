import os
import numpy as np
from PIL import Image

class SARASDataset:
    def __init__(self, data_dir):
        """
        Initialize SARAS dataset loader.

        Args:
            data_dir (str): Directory containing set1/ and set2/ and Obj.names.
        """
        self.data_dir = data_dir
        self.frames = []
        self.class_names = {}

        self._load_class_names()
        self._scan_dataset()

    def _load_class_names(self):
        obj_names_path = os.path.join(self.data_dir, "Obj.names")
        if os.path.exists(obj_names_path):
            with open(obj_names_path, 'r') as f:
                for i, line in enumerate(f):
                    name = line.strip()
                    if name:
                        self.class_names[i] = name
        else:
            print(f"Warning: {obj_names_path} not found. Class names will be integer IDs.")

    def _scan_dataset(self):
        for set_dir in ['set1', 'set2']:
            full_set_dir = os.path.join(self.data_dir, set_dir)
            if not os.path.exists(full_set_dir):
                continue

            for filename in os.listdir(full_set_dir):
                if filename.endswith(".jpg"):
                    base_name = os.path.splitext(filename)[0]
                    txt_path = os.path.join(full_set_dir, f"{base_name}.txt")

                    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                        self.frames.append(os.path.join(full_set_dir, filename))

    def get_frame_data(self, frame_path):
        """
        Load an RGB frame and its bounding boxes.

        Returns:
            frame_np: shape (3, H, W) normalized to float32
            bboxes: list of dicts {'bbox': [cx, cy, w, h], 'label': int, 'label_name': str}
        """
        img = Image.open(frame_path).convert('RGB')

        frame_np = np.array(img).astype(np.float32) / 255.0
        frame_np = np.transpose(frame_np, (2, 0, 1)) # (H, W, 3) to (3, H, W)

        base_name = os.path.splitext(frame_path)[0]
        txt_path = f"{base_name}.txt"

        bboxes = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        label = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        label_name = self.class_names.get(label, str(label))
                        bboxes.append({
                            'bbox': [cx, cy, w, h],
                            'label': label,
                            'label_name': label_name
                        })

        return frame_np, bboxes
