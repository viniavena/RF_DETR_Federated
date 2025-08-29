import os
import json
from PIL import Image

def yolo_to_coco(images_dir, labels_dir, classes, output_json):
    """
    Converte dataset YOLO para COCO.
    
    Args:
        images_dir (str): Diretório com imagens.
        labels_dir (str): Diretório com arquivos .txt YOLO correspondentes.
        classes (list): Lista com nomes das classes (índice = ID da classe YOLO).
        output_json (str): Caminho para salvar o arquivo COCO .json.
    """
    coco_data = {
        "info": {
            "description": "Dataset convertido de YOLO para COCO",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "Meu Nome",
            "date_created": "2025-08-11"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Tipo licença",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Registra as categorias
    for i, class_name in enumerate(classes, start=1):
        coco_data["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory" : "none"
        })

    ann_id = 1
    img_id = 1

    # Adiciona info das imagens
    for img_file in sorted(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, img_file)
    
        if not os.path.isfile(img_path):
            continue

        try:
            width, height = Image.open(img_path).size
            coco_data["images"].append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        except Exception as e:
            print(f"Erro ao abrir imagem {img_path}: {e}")
            continue

        # Pega o arquivo de rotulo YOLO correspondente
        label_file = os.path.splitext(img_file)[0] + ".txt" # troca .jpg ou .png por .txt
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            with open(label_path, "r") as lf:
                for line in lf:
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    # YOLO usa posicao normalizada [0,1], COCO usa pixels absolutos (x_min, y_min, width, height)
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2

                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(class_id),
                        "bbox": [x_min, y_min, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    ann_id += 1

        img_id += 1

    # Salva JSON final
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=4)

    print(f"Arquivo COCO salvo em: {output_json}")


if __name__ == "__main__":
    classes = ['pedestrian']

    yolo_to_coco(
        images_dir="/home/gta/PennFudanYOLO/images/train", # dataset/images/train
        labels_dir="/home/gta/PennFudanYOLO/labels/train", # dataset/labels/train
        classes=classes,
        output_json="train_coco.json"
    )

    yolo_to_coco(
        images_dir="/home/gta/PennFudanYOLO/images/test", # dataset/images/val
        labels_dir="/home/gta/PennFudanYOLO/labels/test", # dataset/labels/val
        classes=classes,
        output_json="test_coco.json"
    )
