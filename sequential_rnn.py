import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NormativeModel import BayesianObserver

# Define the RNN model
class HazardRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=800, output_size=2, num_layers=1):
        super(HazardRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.fc_input = nn.Linear(input_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, output_size)  # Output 2 classes (1 or -1)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.squeeze(-1)  # Shape: [variants, experiments, seq_len, trueVal]

        batch_size, num_experiments, seq_len, _ = x.shape  # Unpack dimensions
        x = x.view(batch_size * num_experiments, seq_len, -1)  # Flatten to [total_batches, seq_len, feature_dim]

        h = torch.zeros(batch_size * num_experiments, self.hidden_size, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            h = self.activation(self.fc_input(x_t) + h)

        out = self.fc_output(h)  # Shape: [batch_size * num_experiments, 2]
        return self.softmax(out)  # Probability distribution over [1, -1]

# Load data function
def load_data(params):
    all_data = []
    for k in range(params['variants']):
        train_data = pd.read_csv(f"variants/trainConfig_var{k}.csv")
        test_data = pd.read_csv(f"variants/testConfig_var{k}.csv")
        
        # Only use the prediction mode for now (ignore report)
        train_data = train_data[train_data.iloc[:, 4] == 'predict']
        test_data = test_data[test_data.iloc[:, 4] == 'predict']

        # Extract evidence and true values directly
        evidence_train = train_data['evidence'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
        true_val_train = train_data['trueVal'].values  # Directly use 'trueVal' column

        evidence_test = test_data['evidence'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
        true_val_test = test_data['trueVal'].values  # Directly use 'trueVal' column

        all_data.append((torch.tensor(np.stack(evidence_train), dtype=torch.float32).unsqueeze(-1),
                         torch.tensor((true_val_train == 1).astype(int), dtype=torch.long)))  # Convert -1/1 to 0/1

        all_data.append((torch.tensor(np.stack(evidence_test), dtype=torch.float32).unsqueeze(-1),
                         torch.tensor((true_val_test == 1).astype(int), dtype=torch.long)))  # Convert -1/1 to 0/1

    return all_data

# Training setup
params = {'variants': 2}
data = load_data(params)
print('Data loaded from file.')

# Unpack training and testing data
X_train, y_train = zip(*data[::2])  # Training data
X_test, y_test = zip(*data[1::2])  # Test data
print('Data processed for classification!')

# Convert to tensors
X_train = torch.stack(X_train).unsqueeze(-1)  # Shape: [5, 800, 20, 1]
y_train = torch.cat(y_train, dim=0)

X_test = torch.stack(X_test).unsqueeze(-1)  # Shape: [5, 800, 20, 1]
y_test = torch.cat(y_test, dim=0)

# Define model, loss, and optimizer
model = HazardRNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)  # Shape: [5 * 800, 2]
    loss = criterion(outputs, y_train)  # y_train shape must be [5 * 800]

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


print("Training complete.")


# Testing loop
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)  # Shape: [5, 800, 2, 1]
    predictions = torch.argmax(test_outputs, dim=-1)  # Get index of highest probability
# Convert labels to match predicted format (0 or 1)
true_labels = (y_test + 1) // 2  # Convert -1,1 → 0,1

# Compare predictions with true labels
correct = (predictions == true_labels).float()  # 1 if correct, 0 otherwise

# Compute sliding average accuracy for the model (window size = 100)
window_size = 50
# Pad with zeros on both sides for centered window
model_sliding_accuracy = F.avg_pool1d(correct.view(1, 1, -1), kernel_size=window_size, stride=1, padding=window_size//2)
model_sliding_accuracy = model_sliding_accuracy.squeeze().numpy()

# Compute cumulative accuracy over time
#model_cumulative_accuracy = correct.cumsum(dim=-1) / (torch.arange(1, correct.size(-1) + 1))

bayesian_cumulative_accuracy = []
bayesian_correct = []

# Run Bayesian Observer on the test set (assuming y_test is the evidence for simplicity)
hs = np.arange(0, 1, 0.05)
y_test_converted = 2 * y_test - 1
for i in range(X_test.size(0) * X_test.size(1)):
    experiment_idx = i // X_test.size(1)
    sample_idx = i % X_test.size(1)

    ev = X_test[experiment_idx, sample_idx, :, 0].numpy().tolist()
    #print(ev)

    _, _, rep, pred = BayesianObserver(ev, mu1=-1, mu2=1, sigma=0.1, hs=hs)
    # print(rep)
    # print(pred)
    # print(y_test_converted[i].item())
    # print('-----')
    #print(bayesian_preds)
    
    # Compare Bayesian predictions to true labels
    bayesian_accuracy = (pred == y_test_converted[i].item())
    bayesian_cumulative_accuracy.append(bayesian_accuracy)
    
    is_correct = (pred == y_test_converted[i].item())
    bayesian_correct.append(is_correct)
    
# Convert to numpy array
bayesian_correct = np.array(bayesian_correct, dtype=np.float32)

bayesian_sliding_accuracy = np.convolve(
    bayesian_correct, np.ones(window_size)/window_size, mode='same'
)

# Convert bayesian_cumulative_accuracy to a cumulative sum
#bayesian_cumulative_accuracy = np.cumsum(bayesian_cumulative_accuracy) / (np.arange(1, len(bayesian_cumulative_accuracy) + 1))

#print(bayesian_cumulative_accuracy)

#plt.plot(model_cumulative_accuracy.numpy(), label="Model", color='blue')
#plt.plot(bayesian_cumulative_accuracy, label="Bayesian Observer", color='green')

plt.plot(model_sliding_accuracy, label="RNN Model", color='blue')
plt.plot(bayesian_sliding_accuracy, label="Bayesian Observer", color='green')
plt.axhline(0.5, color='r', linestyle='--', label="Random Baseline (50%)")

plt.xlabel("Time Step")
plt.ylabel("Cumulative Accuracy")
plt.title("Model Accuracy vs Bayesian Observer Accuracy Over Time")
plt.legend()
plt.show()


num_experiments = 4
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i in range(num_experiments):
    sample_idx = i  
    sample_evidence = X_test[0, sample_idx]  # Shape: (20, 1)
    true_next_drop = y_test[sample_idx].item() * 2 -1  # The actual next drop (-1 or 1)

    sample_evidence = sample_evidence.unsqueeze(0).unsqueeze(0)


    # Get model prediction
    with torch.no_grad():
        output = model(sample_evidence)  # Add batch dim
        predicted_label = torch.argmax(output).item()  # Get class index

    # Convert prediction from index (0 or 1) to (-1 or 1)
    predicted_next_drop = -1 if predicted_label == 0 else 1

    time_steps = list(range(1, 21))

    sample_evidence_flat = sample_evidence.squeeze().numpy().flatten()

    # Select subplot axis
    ax = axes[i // 2, i % 2]

    # Plot the first 20 data points
    ax.plot(time_steps, sample_evidence_flat, 'bo-', label="Observed Data")

    # Plot the 21st time step (True vs Predicted)
    ax.scatter(21, true_next_drop, color='g', marker='o', label="True Next Drop", s=100)
    ax.scatter(21, predicted_next_drop, color='r', marker='x', label="Predicted Next Drop", s=100)

    # Add a horizontal line at y=0 for reference
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)

    # Customize plot
    ax.set_xticks(range(1, 22))
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Data Value / Next Drop")
    ax.set_title(f"Test Experiment {i + 1}: Prediction vs True Next Drop")
    ax.legend()

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()
