import torch
from torch import nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print("fc1 output:", x.dtype)
        print()
        x = self.relu(x)
        x = self.ln(x)
        print("layer norm output:", x.dtype)
        print()
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    device = "cuda"
    net = ToyModel(10, 10).to(device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        print("model params:")
        for name, param in net.named_parameters():
            print(" ", name, param.dtype)
        print()

        x = torch.randn(10, 10).to(device)
        logit = net(x)
        print("logit:", logit.dtype)
        print()

        loss = F.cross_entropy(logit, torch.randint(10, (10,)).to(device))
        print("loss:", loss.dtype)
        print()

        loss.backward()

        print("model grads:")
        for name, param in net.named_parameters():
            print(" ", name, param.grad.dtype)
