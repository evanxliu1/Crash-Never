import torch
from windowed_feature_dataset import WindowedFeatureDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset = WindowedFeatureDataset('windowed_features', 'train.csv', window_size=30)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Total windows: {len(dataset)}")
    for i, (x, y) in enumerate(loader):
        print(f"Batch {i}: x shape = {x.shape}, y = {y}")
        if i == 2:
            break
    # Inspect one sample
    x0, y0 = dataset[0]
    print(f"First window shape: {x0.shape}, label: {y0}")
    # Check label distribution
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    print(f"Label counts: {torch.bincount(torch.tensor(labels))}")
    assert any(l == 1 for l in labels), "No positive labels found! Check label alignment."
