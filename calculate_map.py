import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def get_map(metric, gt_nums):
    # metric: nc x N(conf, tp)
    # gt_nums: nc-gt_num
    mAP = 0
    exist_class = 0
    p = r = 0.0
    for i in range(len(metric)):
        if gt_nums[i] == 0:
            if len(metric[i]) != 0:
                exist_class += 1
            continue
        if len(metric[i]) == 0:
            exist_class += 1
            continue
        single_cls_metric = np.array(metric[i])
        single_cls_metric = single_cls_metric[np.argsort(-single_cls_metric[:, 0])]
        tps = single_cls_metric[:, 1]
        precision = np.zeros(len(tps))
        recall = np.zeros(len(tps))
        tp = 0
        for j in range(len(tps)):
            tp += tps[j]
            precision[j] = tp / (j + 1)
            recall[j] = tp / gt_nums[i]
        for j in range(len(tps) - 2, -1, -1):
            precision[j] = np.max([precision[j], precision[j + 1]])
        p += np.min(precision)
        r += np.max(recall)
        precision = np.concatenate(([1.0], precision, [0.0]))
        recall = np.concatenate(([0.0], recall, [1.0]))

        x = np.linspace(0, 1, 101)  
        AP = np.trapz(np.interp(x, recall, precision), x)
        mAP += AP
        exist_class += 1
    return mAP / exist_class, p / exist_class, r / exist_class

def calculate_iou(boxes1, boxes2):
    box1_len = len(boxes1)
    box2_len = len(boxes2)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    area1 = np.repeat(area1.reshape(box1_len, 1), box2_len, 1)
    area2 = np.repeat(area2.reshape(1, box2_len), box1_len, 0)
    boxes1_xyxy = boxes1[:, 0:4]
    boxes2_xyxy = boxes2[:, 0:4]
    boxes1_xyxy = np.repeat(boxes1_xyxy.reshape(box1_len, 1, 4), box2_len, 1)
    boxes2_xyxy = np.repeat(boxes2_xyxy.reshape(1, box2_len, 4), box1_len, 0)
    lt = np.maximum(boxes1_xyxy[:, :, 0:2], boxes2_xyxy[:, :, 0:2])
    rb = np.minimum(boxes1_xyxy[:, :, 2:4], boxes2_xyxy[:, :, 2:4])
    inter = rb - lt
    inter[inter < 0] = 0
    inter = inter[:, :, 0] * inter[:, :, 1]
    return inter / (area1 + area2 - inter)

def deal_image(preds, gts, metric, threshold):
    # preds: N x (x1, y1, x2, y2, class_id, score)
    # gts: M x (x1, y1, x2, y2, score)
    preds = preds[np.argsort(-preds[:,-1])]

    class_id = preds[:, -2].astype(int)
    cls = np.unique(class_id)
    cls_all = len(metric)
    for c in cls:
        if c >= cls_all:
            continue
        pred_c = preds[class_id == c]
        if len(gts) < 1:
            for index in range(len(pred_c)):
                metric[c].append([pred_c[index][-1], 0.0])
            continue
        gt_c = gts[gts[:, -1] == c]
        if len(gt_c) == 0:
            for index in range(len(pred_c)):
                metric[c].append([pred_c[index][-1], 0.0])
            continue
        ious = calculate_iou(pred_c, gt_c)
        max_area = ious.max(1)
        max_area_index = ious.argmax(1)
        max_area_index[max_area < threshold] = -1
        for i in range(len(max_area_index)):
            if i == 0 or max_area_index[i] == -1:
                continue
            if max_area_index[i] in max_area_index[:i]:
                max_area_index[i] = -1
        max_area_index[max_area_index > -1] = 1
        max_area_index[max_area_index < 0] = 0
        sums_items = np.zeros((len(pred_c), 2))
        sums_items[:, 0] = pred_c[:, -1]
        sums_items[:, 1][max_area_index.astype(bool)] = 1

        for conf, tfp in sums_items:
            metric[c].append([conf, tfp])
    return metric

def get_gt_num(gts, gt_nums):
    if len(gts) == 0:
        return gt_nums
    class_id = gts[:, -1].astype(int)
    cls = np.unique(class_id)
    for c in cls:
        gt = gts[class_id == c]
        gt_nums[c] += len(gt)
    return gt_nums

def get_prediction_info(prediction_path, data_type):
    print("start load predicitons ...")
    prediction_info = {}
    predictions = json.load(open(prediction_path, 'r'))
    for pred in predictions:
        x, y, w, h = pred["bbox"]
        if isinstance(x, str):
            x, y, w, h = list(map(float, [x, y, w, h]))
        x1, y1, x2, y2 = x, y, x + w, y + h
        if data_type == 'yolo':
            category_id = float(pred["category_id"])
        elif data_type == 'regionclip':
            category_id = float(pred["category_id"]) - 1
        if pred["image_id"] not in prediction_info:
            prediction_info[pred["image_id"]] = [[x1, y1, x2, y2]+[category_id, float(pred["score"])]]
        else:
            prediction_info[pred["image_id"]].append([x1, y1, x2, y2]+[category_id, float(pred["score"])])
    print("finish loading predictions!")
    return prediction_info

if __name__ == "__main__":
    image_path = "/workspace/songfei/datasets/objects34/images"
    ann_path = "/workspace/songfei/datasets/objects34/labels_34"
    prediction_path = "/workspace/songfei/YoloV8/zero_shot/detect/val_34_x_224/clip.json"
    num_class = 33
    iou_threshold = 0.5
    iou_threshold_add = 0.05
    data_type = "yolo"
    
    prediction_info = get_prediction_info(prediction_path, data_type)

    gt_nums = [0 for _ in range(num_class)]
    iou_threshold_curr = iou_threshold
    metric = {}
    while iou_threshold_curr < 1:
        metric[iou_threshold_curr] = [[] for _ in range(num_class)]
        iou_threshold_curr += iou_threshold_add
    background = []
    for index, image_name in tqdm(enumerate(os.listdir(image_path))):
        img = Image.open(os.path.join(image_path, image_name))
        if data_type == "yolo":
            try:
                image_id = int(image_name.strip(".jpg"))
            except:
                image_id = image_name.strip(".jpg")
        elif data_type == "regionclip":
            image_id = index
            
        gts = []
        w, h = img.size
        label_file = os.path.join(ann_path, image_name.replace(".jpg", ".txt"))
        with open(label_file, "r") as f:
            for line in f.readlines():
                line = line.strip("\n")
                cls, cx, cy, ow, oh = line.split()

                x1 = (float(cx) - float(ow) / 2) * w
                y1 = (float(cy) - float(oh) / 2) * h
                x2 = (float(cx) + float(ow) / 2) * w
                y2 = (float(cy) + float(oh) / 2) * h

                gts.append([x1, y1, x2, y2, float(cls)])
        gt_nums = get_gt_num(np.array(gts), gt_nums)
        if image_id in prediction_info:
            iou_threshold_curr = iou_threshold
            while iou_threshold_curr < 1:
                metric[iou_threshold_curr] = deal_image(np.array(prediction_info[image_id]), np.array(gts), metric[iou_threshold_curr], iou_threshold_curr)
                iou_threshold_curr += iou_threshold_add
        else:
            background.append(image_name)

    mAP = {}
    iou_threshold_curr = iou_threshold
    while iou_threshold_curr < 1:
        mAP[iou_threshold_curr], p, r = get_map(metric[iou_threshold_curr], gt_nums)
        if iou_threshold_curr == iou_threshold:
            precision = p
            recall = r
        iou_threshold_curr += iou_threshold_add
        
    print("{} backgrounds".format(len(background)))
    # print(background)
    print("mAP@0.5 = {:.3f}".format(mAP[0.5]))
    print("mAP = {:.3f}".format(np.mean(list(mAP.values()))))
    print("Precision = {:.3f}".format(precision))
    print("Recall = {:.3f}".format(recall))
