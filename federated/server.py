import argparse
from datetime import datetime
import flwr as fl

NUM_CLIENTS = 2
NUM_ROUNDS = 10

TRAIN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

def fit_config(server_round: int):
    return {"train_id": TRAIN_ID, "server_round": server_round}

def evaluate_config(server_round: int):
    return {"train_id": TRAIN_ID, "server_round": server_round}

def main():
    parser = argparse.ArgumentParser("Servidor Flower para RF-DETR")
    parser.add_argument("--server_ip", type=str, default="localhost")
    parser.add_argument("--server_port", type=str, default="8080")
    parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--num_rounds", type=int, default=NUM_ROUNDS)
    args = parser.parse_args()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=args.num_clients,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )

    fl.server.start_server(
        server_address=f"{args.server_ip}:{args.server_port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
