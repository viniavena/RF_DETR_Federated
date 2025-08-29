import os
import csv
import torch

def get_weights(rfdetr_model):
    """Extrai os pesos do modelo PyTorch dentro do RFDETR."""
    torch_model = rfdetr_model.model.model  # acessa o torch.nn.Module real
    return [tensor.cpu().numpy() for tensor in torch_model.state_dict().values()]

def set_weights(rfdetr_wrapper, weights):
    torch_model = rfdetr_wrapper.model.model
    local_sd = torch_model.state_dict()
    keys = list(local_sd.keys())
    incoming_sd = {k: torch.tensor(v) for k, v in zip(keys, weights)}
    merged = {}
    for k, v_local in local_sd.items():
        v_glob = incoming_sd.get(k)
        merged[k] = v_glob if (v_glob is not None and tuple(v_glob.shape) == tuple(v_local.shape)) else v_local
    torch_model.load_state_dict(merged, strict=True)

def save_parameters(wrapper, out_path: str):
    torch_module = wrapper.model.model
    torch.save(torch_module.state_dict(), out_path)
    print(f"Model weights saved at {out_path}")


# def load_parameters(model_path: str, device: str):
#     state_dict = torch.load(model_path)
#     model_wrapper.model.load_state_dict(state_dict, strict=False)
#     return model_wrapper

def load_parameters(rfdetr_class, model_path: str, device: torch.device):
    state_dict = torch.load(model_path, map_location=device)

    num_classes = None
    if "class_embed.weight" in state_dict:
        num_classes = state_dict["class_embed.weight"].shape[0]

    rfdetr = rfdetr_class()
    
    if (num_classes is not None) and hasattr(rfdetr.model, "reinitialize_detection_head"):
        rfdetr.model.reinitialize_detection_head(num_classes)

    # 5) aplica os pesos no nn.Module interno
    torch_module = rfdetr.model.model
    torch_module.load_state_dict(state_dict, strict=False)

    try:
        torch_module.to(device) 
        rfdetr.model.device = device
    except Exception:
        pass

    return rfdetr

# Merge de pesos globais
def merge_global_state_dict(model, global_state_dict):
    """Faz merge apenas das chaves compatíveis."""
    full_sd = model.model.state_dict()
    updated_keys = []
    for k in global_state_dict:
        if k in full_sd and full_sd[k].shape == global_state_dict[k].shape:
            full_sd[k] = global_state_dict[k]
            updated_keys.append(k)
    model.model.load_state_dict(full_sd, strict=True)
    return model, updated_keys

# Logging
def log_metrics_to_csv(filename, round_num, client_id, metrics: dict):
    """Salva métricas em CSV."""
    is_new_file = not os.path.exists(filename)
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            header = ["round", "client_id"] + list(metrics.keys())
            writer.writerow(header)
        row = [round_num, client_id] + [metrics[k] for k in metrics]
        writer.writerow(row)