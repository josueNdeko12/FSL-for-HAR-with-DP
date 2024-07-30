import copy
from DP import *

# Function to simulate client-side LSTM processing with optional differential privacy
def client_processing(client_model, train_samples, use_dp=False, epsilon=0.0, delta=0.0):
    train_samples = torch.tensor(train_samples, dtype=torch.float32).to(next(client_model.parameters()).device)
    output = client_model(train_samples)
    if use_dp:
        output = add_gaussian_noise(output, epsilon, delta)
    return output

# Function to simulate server-side LSTM processing
def server_processing(server_model, cut_input):
    return server_model(cut_input)

# Function to aggregate client model weights
def aggregate_client_weights(client_models):
    new_weights = []
    for weights in zip(*[model.state_dict().values() for model in client_models]):
        new_weights.append(torch.mean(torch.stack(list(weights)), dim=0))
    return new_weights

# Function to set the client model weights
def set_client_weights(client_model, new_weights):
    state_dict = client_model.state_dict()
    for key, value in zip(state_dict.keys(), new_weights):
        state_dict[key] = value
    client_model.load_state_dict(state_dict)

# Function to aggregate model weights
def aggregate_weights(models):
    avg_weights = copy.deepcopy(models[0].state_dict())
    for key in avg_weights.keys():
        avg_weights[key] = torch.mean(torch.stack([model.state_dict()[key].float() for model in models]), dim=0)
    return avg_weights

# Function to set the client model weights
def set_model_weights(model, new_weights):
    state_dict = model.state_dict()
    for key, value in zip(state_dict.keys(), new_weights.values()):
        state_dict[key] = value.clone()
    model.load_state_dict(state_dict)
