import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from pathlib import Path


class ModelDiagnostics:
    def __init__(self, model, criterion, output_dir="output/"):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion
        self.output_dir = Path(output_dir)
        Path(self.output_dir).mkdir(exist_ok=True)

    def plot_loss_landscape(self, epoch, val_loader, run_name):
        weights = list(self.model.parameters())
        original_w = weights[0].clone()

        alpha = np.linspace(-1, 1, 20)
        beta = np.linspace(-1, 1, 20)
        z = np.zeros((len(alpha), len(beta)))

        for i, a in enumerate(alpha):
            for j, b in enumerate(beta):
                weights[0].data = (
                    original_w
                    + a * torch.randn_like(original_w)
                    + b * torch.randn_like(original_w)
                )
                input_ids, attention_masks, labels = next(iter(val_loader))
                input_ids, attention_masks, labels = (
                    input_ids.to(self.device),
                    attention_masks.to(self.device),
                    labels.to(self.device),
                )
                outputs = self.model(input_ids, attention_masks=attention_masks)
                loss = self.criterion(outputs, labels)
                z[i, j] = loss.item()

        weights[0].data = original_w  # Reset the weights

        plt.figure(figsize=(10, 8))
        plt.contourf(alpha, beta, z, levels=50)
        plt.colorbar(label="Loss")
        plt.title(f"Loss Landscape at Epoch {epoch}")
        plt.xlabel("α")
        plt.ylabel("β")
        plt.savefig(f"{self.output_dir}/loss_landscape_{run_name}_epoch_{epoch}.png")
        plt.close()

    def check_gradient_flow(self, epoch, run_name):
        """
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        """
        ave_grads = []
        max_grads = []
        layers = []

        for n, p in self.model.named_parameters():
            if p.requires_grad and ("bias" not in n) and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())

        plt.figure(figsize=(10, 8))
        plt.bar(range(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(range(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title(f"Gradient flow at epoch {epoch}")
        plt.grid(True)
        plt.legend(
            [
                Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4),
            ],
            ["max-gradient", "mean-gradient", "zero-gradient"],
        )

        plt.tight_layout()
        plt.savefig(Path(self.output_dir) / f"gradient_flow_epoch_{epoch}.png")
        plt.close()

        Path(self.output_dir / run_name).mkdir(exist_ok=True)
        with open(self.output_dir / run_name / f"epoch-{epoch}", "w") as output:
            print(f"Gradient flow at epoch {epoch}:", file=output)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    print(
                        f"{name}: mean {param.grad.abs().mean().item():.5f}, max {param.grad.abs().max().item():.5f}",
                        file=output,
                    )
