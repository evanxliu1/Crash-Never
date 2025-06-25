import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class WindowedFeatureDataset(Dataset):
    def __init__(self, windowed_root, csv_path, window_size=30, transform=None):
        self.windowed_root = windowed_root
        self.window_files = sorted([f for f in os.listdir(windowed_root) if f.endswith('_windowed.npy')])
        self.video_ids = [f.split('_')[0] for f in self.window_files]
        self.window_sizes = []  # number of windows per video
        self.index_map = []  # (video_idx, window_idx)
        self.transform = transform
        # Load label CSV
        df = pd.read_csv(csv_path, dtype={'id': str})
        df['id'] = df['id'].apply(lambda x: str(x).zfill(5))    
        self.labels = {}  # (video_id, window_idx) -> label
        for i, vid in enumerate(self.video_ids):
            # IDs are now zero-padded in both df and filenames
            video_windows = np.load(os.path.join(windowed_root, self.window_files[i]))
            n_windows = video_windows.shape[0]
            self.window_sizes.append(n_windows)
            # Find label for each window
            meta = df[df['id'] == vid]
            if len(meta) == 0:
                print(f"Warning: no label found for video {vid} (padded: {vid_padded}), defaulting to 0.")
                for widx in range(n_windows):
                    self.labels[(vid, widx)] = 0
                continue
            meta = meta.iloc[0]
            alert_frame = int(meta['time_of_alert'] * meta.get('frame_rate', 30)) if not np.isnan(meta['time_of_alert']) else np.inf
            for widx in range(n_windows):
                window_start = widx * 6  # stride=6
                # If window_start+window_size > alert_frame, label positive
                if window_start + window_size > alert_frame:
                    self.labels[(vid, widx)] = 1
                else:
                    self.labels[(vid, widx)] = 0
            for widx in range(n_windows):
                self.index_map.append((i, widx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        video_idx, window_idx = self.index_map[idx]
        vid = self.video_ids[video_idx]
        arr = np.load(os.path.join(self.windowed_root, self.window_files[video_idx]))
        window = arr[window_idx]  # (window_size, feature_dim)
        label = self.labels[(vid, window_idx)]
        if self.transform:
            window = self.transform(window)
        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Example usage:
# dataset = WindowedFeatureDataset('windowed_features', 'train.csv', window_size=30)
# loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
