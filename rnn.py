import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NormativeModel import BayesianObserver

class HazardRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=5000, output_size=20, num_layers=1):
        super(HazardRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.fc_input = nn.Linear(input_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, output_size)  
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size) 
        
        for t in range(seq_len):
            x_t = x[:, t, :] 
            h = self.activation(self.fc_input(x_t) + h)
        out = self.fc_output(h)
        return self.softmax(out)

# Load test and training data (all variants)
def load_data(params):
    all_data = []
    for k in range(0, params['variants']):
        train_data = pd.read_csv(f"variants/trainConfig_var{k}.csv")
        test_data = pd.read_csv(f"variants/testConfig_var{k}.csv")

        evidence_train = train_data['evidence'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
        hazards_train = train_data['trueHazard'].values

        evidence_test = test_data['evidence'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
        hazards_test = test_data['trueHazard'].values

        all_data.append((torch.tensor(np.stack(evidence_train), dtype=torch.float32).unsqueeze(-1), torch.tensor(hazards_train, dtype=torch.float32).unsqueeze(-1)))
        all_data.append((torch.tensor(np.stack(evidence_test), dtype=torch.float32).unsqueeze(-1), torch.tensor(hazards_test, dtype=torch.float32).unsqueeze(-1)))
    
    return all_data

def getBayesianData(data, hs, mu1, mu2, sigma):
    processed_data = []
    for evidence, true_hazard in data:
        np_evidence = evidence.squeeze(-1).numpy()  # Shape (800, 20)
        
        for trial in np_evidence:  # Iterate over each trial (shape (20,))
            L_haz, _, _, _ = BayesianObserver(trial, mu1, mu2, sigma, hs)  # Compute hazard beliefs
            hazard_beliefs = L_haz[:, -1]  # Final posterior belief over hazard rates
            label = torch.tensor(hazard_beliefs, dtype=torch.float32)  # Convert to tensor
            processed_data.append((torch.tensor(trial, dtype=torch.float32), label))

    return processed_data

# Training loop
params = {}
params['variants']=20
data = load_data(params)
print('Data loaded from file.')
hs = np.arange(0, 1, 0.05)  # Hazard rate bins
X_train, y_train = zip(*getBayesianData(data[::2], hs, mu1=-1, mu2=1, sigma=0.1))
X_test, y_test = zip(*getBayesianData(data[1::2], hs, mu1=-1, mu2=1, sigma=0.1))
print('Bayesians computed!')

# Convert to tensors
X_train = torch.stack(X_train)
y_train = torch.stack(y_train)
X_test = torch.stack(X_test)
y_test = torch.stack(y_test)

X_train = X_train.unsqueeze(-1)
X_test = X_test.unsqueeze(-1)


# Define model, loss, and optimizer
model = HazardRNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Testing loop
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

print("Training complete.")


# Example trial visualization
sample_idx = 0
sample_evidence = X_test[sample_idx]
#print(sample_evidence)
predicted_hazard = []
for t in range(len(sample_evidence)):
    input_frame = sample_evidence[t].unsqueeze(0).unsqueeze(0)
    #print(input_frame)
    output = model(input_frame).squeeze().detach().numpy()
    predicted_hazard.append(output)

plt.plot(hs, predicted_hazard[-1], label="Predicted Hazard Belief", color='blue')
plt.plot(hs, y_test[sample_idx].numpy(), label="True Bayesian Belief", color='red', linestyle='--')
plt.xlabel("Hazard Rate")
plt.ylabel("Probability")
plt.title("Hazard Belief Distribution (Final Time Step)")
plt.legend()
plt.show()
