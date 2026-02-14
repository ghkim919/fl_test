import torch
import flwr as fl
from flwr.client import ClientApp
from flwr.common import Context

from src.model import create_model, get_parameters, set_parameters, train, test
from src.dataset import init_fds, load_partition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        epochs = int(config.get("local_epochs", 3))
        lr = float(config.get("learning_rate", 0.01))
        proximal_mu = float(config.get("proximal_mu", 0.0))

        global_params = None
        if proximal_mu > 0.0:
            global_params = [p.clone().detach() for p in self.model.parameters()]

        train(
            self.model,
            self.train_loader,
            epochs,
            lr,
            DEVICE,
            proximal_mu,
            global_params,
        )

        return (
            get_parameters(self.model),
            len(self.train_loader.dataset),
            {},
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.val_loader, DEVICE)
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}


def create_client_app(config):
    num_clients = config["num_clients"]
    partition_config = dict(config["partition"])
    seed = config["seed"]
    batch_size = config["batch_size"]
    model_name = config.get("model", "cnn")

    def client_fn(context: Context):
        init_fds(num_clients, partition_config, seed)
        partition_id = context.node_config["partition-id"]
        train_loader, val_loader = load_partition(
            partition_id,
            batch_size=batch_size,
            seed=seed,
        )
        model = create_model(model_name)
        return FlowerClient(model, train_loader, val_loader).to_client()

    return ClientApp(client_fn=client_fn)
