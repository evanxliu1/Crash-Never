import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from windowed_feature_dataset import WindowedFeatureDataset
from lstm_sequence_model import LSTMSequenceModel

if __name__ == "__main__":
    # Hyperparameters
    input_dim = 525
    hidden_dim = 128
    num_layers = 1
    num_classes = 2
    batch_size = 32
    num_epochs = 5  # minimal run
    lr = 1e-3

    # Dataset and train/val split
    dataset = WindowedFeatureDataset('windowed_features', 'train.csv', window_size=30)
    n_total = len(dataset)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Analyze class distribution
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    from collections import Counter
    class_counts = Counter(labels)
    print(f"Class distribution: {class_counts}")
    weights = [1.0 / class_counts[cls] for cls in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # Model, loss, optimizer
    model = LSTMSequenceModel(input_dim, hidden_dim, num_layers=2, num_classes=num_classes, dropout=0.3, bidirectional=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early stopping and checkpointing
    best_val_roc = 0
    patience = 2
    patience_counter = 0
    best_model_path = 'best_lstm_model.pt'

    from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
    import numpy as np

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        for x, y in train_loader:
            # Normalize features per batch
            x = (x - x.mean(dim=(0,1), keepdim=True)) / (x.std(dim=(0,1), keepdim=True) + 1e-6)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
        train_preds = np.concatenate(all_preds)
        train_targets = np.concatenate(all_targets)
        train_acc = (train_preds == train_targets).mean()
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_set):.4f}, train_acc={train_acc:.4f}")
        # Train metrics
        train_cm = confusion_matrix(train_targets, train_preds)
        print(f"Train confusion matrix:\n{train_cm}")
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(train_targets, train_preds, average='binary', zero_division=0)
        print(f"Train precision: {train_prec:.3f}, recall: {train_rec:.3f}, f1: {train_f1:.3f}")

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_probs = []
        with torch.no_grad():
            for x, y in val_loader:
                x = (x - x.mean(dim=(0,1), keepdim=True)) / (x.std(dim=(0,1), keepdim=True) + 1e-6)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[:,1]
                preds = logits.argmax(dim=1)
                val_preds.append(preds.cpu().numpy())
                val_targets.append(y.cpu().numpy())
                val_probs.append(probs.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_probs = np.concatenate(val_probs)
        val_acc = (val_preds == val_targets).mean()
        val_cm = confusion_matrix(val_targets, val_preds)
        print(f"Epoch {epoch+1}: val_acc={val_acc:.4f}")
        print(f"Val confusion matrix:\n{val_cm}")
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='binary', zero_division=0)
        print(f"Val precision: {val_prec:.3f}, recall: {val_rec:.3f}, f1: {val_f1:.3f}")
        try:
            val_roc = roc_auc_score(val_targets, val_probs)
            print(f"Val ROC-AUC: {val_roc:.3f}")
        except Exception as e:
            val_roc = 0
            print(f"Val ROC-AUC: N/A ({e})")
        # Early stopping & checkpoint
        if val_roc > best_val_roc:
            best_val_roc = val_roc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved (epoch {epoch+1})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training with advanced metrics, class weighting, and regularization complete.")
