import json
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Dict, List, Set

from comet_ml import Experiment
import clip
import torch
from torch import optim, nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CIRRDataset, ImageNetDataset
from phi import ExtendedPhi
from utils import collate_fn, device, contrastive_loss


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path) -> None:
    """Save the weights of the model during training"""
    models_path = training_path / "checkpoints"
    models_path.mkdir(exist_ok=True, parents=True)
    torch.save({
        'epoch': cur_epoch,
        'model_state_dict': model_to_save.state_dict(),
    }, str(models_path / f'{name}.pt'))


def train_one_epoch(loader: DataLoader, model: ExtendedPhi, optimizer: optim.Optimizer, scaler: GradScaler, criterion, epoch: int):
    """Train the model for one epoch"""
    model.train()
    epoch_loss = 0
    train_bar = tqdm(loader, desc=f"Training Epoch {epoch}")

    for batch in train_bar:
        images = batch['reference_image'].to(device)
        text_features = torch.randn(images.size(0), 512).to(device)  # Dummy text features for example

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images, text_features)
            loss = criterion(images, outputs, torch.ones(images.size(0)).to(device))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        train_bar.set_postfix(loss=f"{epoch_loss:.4f}")

    return epoch_loss / len(loader)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save models")
    args = parser.parse_args()

    # Initialize dataset and dataloader
    dataset = CIRRDataset(dataset_path=args.dataset_path, split="train", mode="relative", preprocess=None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = ExtendedPhi(input_dim=512, hidden_dim=1024, output_dim=512, dropout=0.1, num_heads=4).to(device)

    # Optimizer, loss, and scaler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = nn.CosineEmbeddingLoss()
    scaler = GradScaler()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(dataloader, model, optimizer, scaler, criterion, epoch)
        print(f"Epoch {epoch} completed with average loss: {epoch_loss:.4f}")

        # Save the model checkpoint
        save_model(name=f"phi_epoch_{epoch}", cur_epoch=epoch, model_to_save=model, training_path=Path(args.output_dir))


if __name__ == "__main__":
    main()
