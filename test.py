import os
import clip
import torch

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from coco_dataset import coco_crop_dataset

def get_dataset(dataset_path):
    image_path = []
    image_class = []
    classes = []
    for index, cls in enumerate(os.listdir(dataset_path)):
        cls_path = os.path.join(dataset_path,cls)
        classes.append(cls)
        for image_name in os.listdir(cls_path):
            image_path.append(os.path.join(cls_path, image_name))
            image_class.append(index)
    return image_path, image_class, classes


if __name__ == "__main__":

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('/root/autodl-tmp/save_models/clip/ViT-B-32.pt', device)

    # Load the dataset
    root = os.path.expanduser("~/.cache")
    dataset_path = "/root/autodl-tmp/datasets/flower_photos"
    image_path, image_class, classes = get_dataset(dataset_path)
    prompt = ["a" + cls for cls in classes]
    coco_dataset = coco_crop_dataset(image_path, image_class, transform=preprocess)
    # train = CIFAR100(root, download=True, train=True, transform=preprocess)
    # test = CIFAR100(root, download=True, train=False, transform=preprocess)


    def val(dataset, text):
        all_predictions = []
        all_labels = []
        bs = 100
        text_inputs = torch.cat([clip.tokenize(text)]*bs).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            for images, labels in tqdm(DataLoader(dataset, batch_size=bs)):
                image_features = model.encode_image(images.to(device))
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = torch.topk(similarity, k=1, dim=-1)

                all_predictions.append(indices.transpose(0, 1).squeeze(0))
                all_labels.append(labels)

        return torch.cat(all_predictions).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    # Calculate the image features
    predictions, labels = val(coco_dataset, prompt)

    # Perform logistic regression

    # Evaluate using the logistic regression classifier
    accuracy = np.mean((labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")