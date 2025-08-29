import os
import json
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

def test(model_obj, val_images_dir, val_annotations_path, conf_thres=0.0, max_images=None):
    """
    Avalia um modelo RF-DETR num conjunto de validação COCO.

    :param model_obj: instância do modelo RFDETRNano (ou similar).
    :param val_images_dir: caminho para o diretório de imagens de validação.
    :param val_annotations_path: caminho para o arquivo JSON COCO de anotações da validação.
    :param conf_thres: confiança mínima para considerar uma detecção.
    :param max_images: se definido, limita o número de imagens avaliadas.
    :return: dict com métricas mAP50-95 e mAP50.
    """
    coco_gt = COCO(val_annotations_path)

    # id -> nome de classe do COCO
    cat_id_to_name = {c['id']: c['name'] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    # nome -> id
    cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}

    coco_dets = []
    img_ids = coco_gt.getImgIds()
    if max_images:
        img_ids = img_ids[:max_images]

    for img_id in tqdm(img_ids, desc="Avaliando"):
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = os.path.join(val_images_dir, img_info['file_name'])
        if not os.path.isfile(img_path):
            continue

        det = model_obj.predict(img_path, threshold = 0)
        if det.xyxy is None:
            continue

        for box, score, cls_id in zip(det.xyxy, det.confidence, det.class_id):
            if score < conf_thres:
                continue
            bbox_xywh = xyxy_to_xywh(box)
            # aqui assumimos que model_obj.class_id está alinhado aos nomes COCO
            cls_name = model_obj.names[int(cls_id)] if hasattr(model_obj, 'names') else str(int(cls_id))
            category_id = cat_name_to_id.get(cls_name, int(cls_id))
            coco_dets.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": bbox_xywh,
                "score": float(score)
            })

    if not coco_dets:
        raise RuntimeError("Nenhuma detecção gerada pelo modelo.")

    coco_dt = coco_gt.loadRes(coco_dets)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP50-95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
    }