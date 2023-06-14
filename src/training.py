# model.py
from collections import Tuple
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from enum import Enum


class DatasetType(Enum):
    HMM = "HMM"
    VIRAL = "viral"


class Config:
    def __init__(self):
        self.dataset: DatasetType = DatasetType.HMM
        self.model_name: str = 'gpt2'
        self.file_path: str = "./data/viral.1.1.genomic.fna"
        self.sequence_length: int = 1000
        self.stride: int = 1000
        self.split_ratio: float = 0.5
        self.substrings_per_seq: int = 20
        self.sparsity: float = 1.1
        self.num_hidden_states: int = 100
        self.num_seqs: int = 100
        self.sequences_shuffle: bool = True
        self.train_bs: int = 32
        self.val_bs: int = 64
        self.n_embed: int = 512
        self.n_layer: int = 4
        self.n_head: int = 16
        self.lr: float = 1e-4
        self.weight_decay: float = 0.00
        self.num_epochs: int = 200
        self.early_stopping_patience: int = 5
        self.print_every: int = 20


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader, device: torch.device, description: str) -> float:
    model.train()
    running_loss = []
    bar = tqdm(train_loader, desc=description)
    for inputs, targets in bar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, labels=targets)
        loss = outputs.loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        bar.set_postfix({"Train Loss": loss.item()})
    return sum(running_loss) / len(running_loss)


def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, description: str) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_acc = 0.0
    bar = tqdm(val_loader, desc=description)
    with torch.no_grad():
        for inputs, targets in bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs, labels=targets)
            val_loss = outputs.loss.mean()
            predictions = outputs.logits.argmax(dim=-1)
            accuracy = (predictions == targets).cpu().float().mean().item()
            total_loss += val_loss.item() * inputs.size(0)
            total_count += inputs.size(0)
            total_acc += accuracy
            bar.set_postfix({"Val Loss": val_loss.item(), "Val Accuracy": accuracy})
    avg_val_loss = total_loss / total_count
    avg_accuracy = total_acc / len(val_loader)
    return avg_val_loss, avg_accuracy


def train_loop(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, config: Config) -> None:
    best_val_loss = float('inf')
    num_epochs_no_improve = 0  # Number of epochs with no improvement in validation loss

    try:
        for epoch in range(config.num_epochs):
            train_loss = train(model, optimizer, train_loader, device, f"Epoch {epoch + 1}/{config.num_epochs} | Training")
            val_loss, val_acc = evaluate(model, val_loader, device, f"Epoch {epoch + 1}/{config.num_epochs} | Validation")
            samples = (epoch+1) * len(train_loader) * config.train_bs
            print(f"Epoch {epoch + 1}/{config.num_epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Accuracy: {val_acc:.5f}")
            wandb.log({"epoch": epoch + 1, "samples": samples, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc})
            
            if (epoch + 1) % 10 == 0:
                # Save the model weights as an artifact every 10 epochs
                artifact = wandb.Artifact(f"model_weights", type='model')
                torch.save(model.state_dict(), 'gpt2_dna.pth')
                artifact.add_file('gpt2_dna.pth')
                wandb.log_artifact(artifact)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                num_epochs_no_improve = 0
            else:
                num_epochs_no_improve += 1
                if num_epochs_no_improve >= config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}...")
                    break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        torch.save(model.state_dict(), 'gpt2_dna.pth')
        wandb.finish()
