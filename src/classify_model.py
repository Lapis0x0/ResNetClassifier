# src/classify_model.py

import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18

def get_classes(main_folder): # 获取主文件夹下的所有子文件夹名称，并按字母顺序排序。
    classes = [folder for folder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder))]
    classes.sort()
    return classes

def setup_output_folders(output_folder, classes):
    os.makedirs(output_folder, exist_ok=True)
    for class_name in classes:
        os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

def load_model(model_path, num_classes):
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def classify_and_save_images(folder_path, model, transform, classes, output_folder):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0)

            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

            predicted_class = classes[predicted.item()]
            destination_folder = os.path.join(output_folder, predicted_class)
            shutil.copy(image_path, destination_folder)
            print(f'Image: {filename}, Predicted: {predicted_class}, Saved to: {destination_folder}')

def run_classification(dataset_path, model_path, test_folder, output_folder):
    classes = get_classes(dataset_path)
    print("识别到的类别标签:")
    for cls in classes:
        print(cls)

    setup_output_folders(output_folder, classes)

    model = load_model(model_path, len(classes))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    classify_and_save_images(test_folder, model, transform, classes, output_folder)
    print('Classification and saving complete.')

if __name__ == "__main__":
    run_classification('data/dataset', 'data/models/model.pth', 'data/需要分类的图片', 'data/分类结果')