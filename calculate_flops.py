from typing import Tuple, Union
import clip
import torch
from clip.model import CLIP
from torch import nn
from thop import profile
from ultralytics import YOLO
from ultralytics.yolo.utils.torch_utils import model_info

def get_class(concept_path):
    classes = {}
    with open(concept_path, "r") as f:
        for index, line in enumerate(f.readlines()):
            classes[line.strip("\n")] = index
    return classes
    

if __name__ == "__main__":

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detection_model = YOLO("/workspace/songfei/YoloV8/zero_shot/detect/train_coco(object)_x/weights/best.pt")
    classify_model, _ = clip.load('/workspace/songfei/pretrained_models/clip/ViT-L-14.pt', device)

    # Calculate flops and params
    model_info(detection_model.model, imgsz=224)

    concept_path = "/workspace/songfei/YoloV8/concepts_70.txt"
    classes = get_class(concept_path)
    prompt = ["a photo of " + cls for cls, _ in classes.items()]
    # prompt = ["a dog"]
    input_image = torch.empty((1, 3, 224, 224), device=device)
    input_text = clip.tokenize(prompt).to(device)
    flops_all, params_all = profile(classify_model, inputs=[input_image, input_text])
    flops_image, params_image = profile(classify_model.visual, inputs=[input_image.type(classify_model.dtype)])
    # flops, params = profile(classify_model_text, inputs=[input_text])
    print("all: flops = {}GFlops, params = {}M".format(flops_all/1e9, params_all/1e6))
    print("image encoder: flops = {}GFlops, params = {}M".format(flops_image/1e9, params_image/1e6))
    
