import copy
import time
import matplotlib.pyplot as plt
import urllib.request
from numpy import std, dstack
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from loadData import *
from utils import *
from models import *

# Adjustable parameters
learning_rate = 0.01
clients = 10
rounds = 100
epsilon = 3 # Epsilon for noise addition
delta = 1e-5  # Delta for noise addition
clip_norm = 10.0  # Clipping norm for gradient clipping
use_dp = False # Boolean to control the use of differential privacy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to train the Split Learning model with federated aggregation and optional differential privacy
def train_fsl(train_x, train_y, test_x, test_y, learning_rate, clients, rounds, epsilon, delta, clip_norm, use_dp):
    verbose, epochs, batch_size = 0, 1, 64  # Training one epoch per round
    n_samples, n_timesteps, n_features = train_x.shape
    n_outputs = train_y.shape[1]

    # Create and compile global client model and server model
    global_client_model = ClientLSTM((n_timesteps, n_features)).to(device)
    global_server_model = ServerLSTM(100, n_outputs).to(device)  # Fixed to match LSTM hidden size

    # Split data among clients
    client_data = np.array_split(train_x, clients)
    client_labels = np.array_split(train_y, clients)

    history = {'loss': [], 'accuracy': [], 'time': []}

    for round in range(rounds):
        print(f'Round {round + 1}/{rounds}')
        start_time = time.time()  # Start timing

        # Create and train individual client and server models
        client_models = [ClientLSTM((n_timesteps, n_features)).to(device) for _ in range(clients)]
        server_models = [ServerLSTM(100, n_outputs).to(device) for _ in range(clients)]
        client_optimizers = [optim.Adam(client_model.parameters(), lr=learning_rate) for client_model in client_models]
        server_optimizers = [optim.Adam(server_model.parameters(), lr=learning_rate) for server_model in server_models]

        for i, (client_model, server_model) in enumerate(zip(client_models, server_models)):
            client_model.load_state_dict(global_client_model.state_dict())  # Initialize with global client weights
            server_model.load_state_dict(global_server_model.state_dict())  # Initialize with global server weights
            client_optimizer = client_optimizers[i]
            server_optimizer = server_optimizers[i]

            client_output = client_processing(client_model, client_data[i], use_dp=use_dp, epsilon=epsilon, delta=delta)
            client_output = client_output.detach().requires_grad_(True)  # Ensure gradients can be computed

            dataset = TensorDataset(client_output, client_labels[i])
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for batch_output, batch_label in dataloader:
                batch_output.retain_grad()  # Ensure we can compute gradients with respect to this tensor

                server_optimizer.zero_grad()
                client_optimizer.zero_grad()

                # Forward pass on server model
                output = server_model(batch_output)
                loss = F.cross_entropy(output, batch_label.argmax(dim=1))

                # Backward pass on server model
                loss.backward(retain_graph=True)
                server_optimizer.step()

                # Get the gradient of the activations with respect to the loss
                grad_activations = batch_output.grad

                # Ensure grad_activations has the correct shape
                if grad_activations is None:
                    grad_activations = torch.ones_like(batch_output)

                # Backward pass on client model using the received gradient
                torch.autograd.backward(batch_output, grad_tensors=grad_activations)
                client_optimizer.step()

        # Aggregate client and server model weights
        new_client_weights = aggregate_weights(client_models)
        new_server_weights = aggregate_weights(server_models)
        set_model_weights(global_client_model, new_client_weights)  # Update global client model with aggregated weights
        set_model_weights(global_server_model, new_server_weights)  # Update global server model with aggregated weights

        # Evaluate model
        test_cut_output = client_processing(global_client_model, test_x, use_dp=False)
        with torch.no_grad():
            output = global_server_model(test_cut_output)
            loss = F.cross_entropy(output, test_y.argmax(dim=1))
            accuracy = (output.argmax(dim=1) == test_y.argmax(dim=1)).float().mean().item()
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy)

        end_time = time.time()  # End timing
        round_time = end_time - start_time
        history['time'].append(round_time)

        print(f'Loss after round {round + 1}: {loss:.4f}')
        print(f'Accuracy after round {round + 1}: {accuracy * 100.0:.2f}%')
        print(f'Time for round {round + 1}: {round_time:.2f} seconds')

    return global_client_model, global_server_model, history

# Function to print the accuracy, loss, and time as comma-separated lists
def print_accuracy_loss_time_lists(history):
    accuracy_list = ', '.join([f"{acc * 100.0:.2f}" for acc in history['accuracy']])
    loss_list = ', '.join([f"{loss:.4f}" for loss in history['loss']])
    time_list = ', '.join([f"{time:.2f}" for time in history['time']])

    print(f"Accuracy per round (%): {accuracy_list}")
    print(f"Loss per round: {loss_list}")
    print(f"Time per round (seconds): {time_list}")

# Function to plot the confusion matrix
def plot_confusion_matrix(client_model, server_model, test_x, test_y):
    activity_labels = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']
    test_cut_output = client_processing(client_model, test_x)
    with torch.no_grad():
        predictions = server_processing(server_model, test_cut_output)
        predicted_classes = predictions.argmax(dim=1)
        true_classes = test_y.argmax(dim=1)
    cm = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=activity_labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.title('Confusion Matrix')
    plt.show()

# Function to plot the accuracy, loss, and time
def plot_accuracy_loss_time(history):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(np.array(history['accuracy']) * 100, label='Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy per Round')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['loss'], label='Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Model Loss per Round')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['time'], label='Time')
    plt.xlabel('Round')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Round')
    plt.legend()

    plt.tight_layout()
    plt.show()

#Start the training process
download_and_extract_dataset(dataset_url, dataset_dir, dataset_zip)
train_x, train_y, test_x, test_y = load_dataset(dataset_dir + '/')

client_model, server_model, history = train_fsl(train_x, train_y, test_x, test_y,
                                                            learning_rate=learning_rate, clients=clients,
                                                            rounds=rounds, epsilon=epsilon, delta=delta,
                                                            clip_norm=clip_norm, use_dp=use_dp)
plot_confusion_matrix(client_model, server_model, test_x, test_y)
print_accuracy_loss_time_lists(history)
plot_accuracy_loss_time(history)
