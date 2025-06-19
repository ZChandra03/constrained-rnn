import torch
import torch.nn as nn

# Define the RNN model
class HazardRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=800, num_bins=2, num_layers=1):
        super(HazardRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_bins = num_bins

        self.fc_input = nn.Linear(input_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, num_bins)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_all: bool = False):
        """
        Accept either
            • x  ∈ ℝ[V, B, T, 1]   (variants × batch × seq_len × feature)
            • x  ∈ ℝ[B, T, 1]      (batch    × seq_len × feature)

        and flatten it to [total_trials, T, 1] before the recurrent loop.
        """
        # ---- reshape to [N, T, 1] ---------------------------------------------
        if x.dim() == 4:                   # [V, B, T, 1]
            V, B, T, F = x.shape
            x = x.view(V * B, T, F)        # [N = V·B, T, 1]
        elif x.dim() == 3:                 # [B, T, 1]  ← already flat
            N, T, F = x.shape
        else:
            raise ValueError(
                f"expected 3-D or 4-D input, got shape {tuple(x.shape)}"
            )

        # ---- recurrent computation --------------------------------------------
        h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        outputs = []

        for t in range(x.size(1)):         # iterate over timesteps
            x_t = x[:, t, :]               # [N, 1]
            h   = self.activation(self.fc_input(x_t) + h)
            out = self.fc_output(h)        # [N, num_bins]
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)  # [N, T, num_bins]

        if return_all:
            return self.softmax(outputs)       # full trajectory
        else:
            return self.softmax(outputs[:, -1, :])  # only final step

