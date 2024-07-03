import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18

# 定义类别标签
classes = ['a.美少女', 'b.截图与杂图', 'c.动物', 'd.漫画']  # 根据你的实际类别名称修改

# 创建对应的类别文件夹
output_folder = '分类结果'  # 保存分类结果的主文件夹
os.makedirs(output_folder, exist_ok=True)
for class_name in classes:
    os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

# 加载模型
model = resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 批量分类并保存到对应文件夹函数
def classify_and_save_images(folder_path, model, transform, classes, output_folder):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # 支持的图像格式
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0)  # 添加批次维度

            # 进行预测
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

            # 获取预测类别
            predicted_class = classes[predicted.item()]

            # 保存图片到对应类别文件夹
            destination_folder = os.path.join(output_folder, predicted_class)
            shutil.copy(image_path, destination_folder)
            print(f'Image: {filename}, Predicted: {predicted_class}, Saved to: {destination_folder}')

# 设置文件夹路径
test_folder = 'xxx/需要分类的图片'  # 替换为你的“测试”文件夹路径

# 调用分类并保存函数
classify_and_save_images(test_folder, model, transform, classes, output_folder)

print('Classification and saving complete.')
