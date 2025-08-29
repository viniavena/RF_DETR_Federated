#!/bin/bash

# Caminhos
BASE_WEIGHTS="/home/gta/avena/rf_detr_train/federated/rf-detr-nano.pth"

# Dataset único com splits train/valid/test (cada um tem _annotations.coco.json)
DATASET_ROOT="/home/gta/avena/rf_detr_train/dataset"
CENTRAL_VAL_DIR="$DATASET_ROOT/valid"
CENTRAL_VAL_ANN="$DATASET_ROOT/valid/_annotations.coco.json"

LOCAL_EPOCHS=5

# Servidor
SERVER_IP="127.0.0.1"
SERVER_PORT="8081"

# Lista de clientes (IDs distintos)
CLIENT_IDS="1 2"

# Rodar o servidor
echo "[INFO] Iniciando servidor..."
python3 server.py \
  --server_ip "$SERVER_IP" \
  --server_port "$SERVER_PORT" \
  --num_clients "$(echo $CLIENT_IDS | wc -w)" \
  --num_rounds 10 &

# Espera um pouco para o servidor subir
sleep 5

# Rodar os clientes em paralelo, todos apontando para o mesmo dataset
for CLIENT_ID in $CLIENT_IDS
do
  echo "[INFO] Iniciando cliente $CLIENT_ID..."
  python3 client.py \
    --base_weights "$BASE_WEIGHTS" \
    --clients_data_dir "$DATASET_ROOT" \
    --centralized_val_dir "$CENTRAL_VAL_DIR" \
    --centralized_val_ann "$CENTRAL_VAL_ANN" \
    --local_epochs "$LOCAL_EPOCHS" \
    --server_ip "$SERVER_IP" \
    --server_port "$SERVER_PORT" &
done

# Esperar todos os processos terminarem
wait
echo "[INFO] Treinamento federado finalizado."
