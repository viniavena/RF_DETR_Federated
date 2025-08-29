import argparse
import os
import tempfile
import logging
from collections import OrderedDict

import numpy as np
import torch
import flwr as fl

import uuid

from rfdetr import RFDETRNano
from utils import (
    get_weights, set_weights, save_parameters, load_parameters,
    merge_global_state_dict, log_metrics_to_csv
)
from test import test

# 1) Cliente Flower customizado para RF-DETR
class RFDETRClient(fl.client.NumPyClient):
    def __init__(
        self,
        base_weights_path: str,
        clients_data_dir: str,
        centralized_val_dir: str,
        centralized_val_ann: str,
        local_epochs: int,
        device: torch.device,
    ):
        self.client_id = str(uuid.uuid4())  # dentro do __init__
        self.base_weights_path = base_weights_path
        self.clients_data_dir = clients_data_dir
        self.centralized_val_dir = centralized_val_dir
        self.centralized_val_ann = centralized_val_ann
        self.local_epochs = local_epochs
        self.device = device

        self.logger = logging.getLogger(f"RFDETRClient_{self.client_id}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.logger.addHandler(ch)

        self.model: RFDETRNano | None = None
        self.train_id: str | None = None

    def get_properties(self, config):
        return {"client_id": str(self.client_id)}

    def get_parameters(self, config):
        if self.model is None:
            self.model = RFDETRNano(pretrain_weights=self.base_weights_path)
            self.model.model.model.to(self.device)
            self.model.model.device = self.device
            
        return get_weights(self.model)


    def set_parameters(self, parameters):
        set_weights(self.model, parameters)
    
    def fit(self, parameters, config):
        self.train_id = config.get("train_id", "default")
        model_filename = f"client_{self.client_id}_{self.train_id}.pth"

        # Carrega modelo existente ou inicializa
        if os.path.exists(model_filename):
            self.model = load_parameters(RFDETRNano, model_filename, self.device)
        else:
            self.model = RFDETRNano(pretrain_weights=self.base_weights_path)

        if parameters is not None:
            set_weights(self.model, parameters)

        print
        self.model.model.model.to(self.device)
        self.model.model.device = self.device

        with tempfile.TemporaryDirectory() as tmpdir:
            round_num = config.get("server_round", -1)
            csv_filename = f"metrics_results_{self.train_id}.csv"

            # 2) Treinamento local
            self.model.train(
                dataset_dir=self.clients_data_dir,
                epochs=self.local_epochs,
                batch_size=4,
                grad_accum_steps=4,
                lr=1e-4,
                output_dir=tmpdir,
                device=self.device
            )

            # 3) Avaliação após treino local
            local_metrics = test(
                model_obj=self.model,
                val_images_dir=self.centralized_val_dir,
                val_annotations_path=self.centralized_val_ann
            )
            log_metrics_to_csv(csv_filename, round_num, self.client_id, local_metrics)

        # Salva checkpoint local
        save_parameters(self.model, model_filename)

        # Prepara saída para o servidor
        weights_prime = get_weights(self.model)

        # Conta amostras do cliente
        num_samples = sum(
            1
            for root, _, files in os.walk(self.clients_data_dir)
            for fn in files
            if fn.lower().endswith((".jpg", ".png", ".jpeg"))
        )

        return weights_prime, num_samples, local_metrics

def main():
    parser = argparse.ArgumentParser("Cliente Flower para RF-DETR")
    parser.add_argument("--base_weights", type=str, default='/home/gta/avena/rf_detr_train/trained_model/checkpoint_best_regular.pth')
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--clients_data_dir", type=str, default='/path/para/dados_cliente')
    parser.add_argument("--centralized_val_dir", type=str, default='/path/para/val/images')
    parser.add_argument("--centralized_val_ann", type=str, default='/path/para/annotations/instances_val.json')
    parser.add_argument("--server_ip", type=str, default='localhost')
    parser.add_argument("--server_port", type=str, default='8080')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Client RF-DETR] usando {device}")

    client = RFDETRClient(
        base_weights_path=args.base_weights,
        clients_data_dir=args.clients_data_dir,
        centralized_val_dir=args.centralized_val_dir,
        centralized_val_ann=args.centralized_val_ann,
        local_epochs=args.local_epochs,
        device=device,
    )

    fl.client.start_numpy_client(
        server_address=f"{args.server_ip}:{args.server_port}",
        client=client,
    )

if __name__ == "__main__":
    main()
