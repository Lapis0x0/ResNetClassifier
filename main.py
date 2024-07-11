# main.py

from src.train_model import train_model
from src.classify_model import run_classification
import os


def create_data_directories(): # 检测是否存在data文件夹，若不存在则创建data及其子文件夹

    base_dir = os.getcwd()  # 获取当前工作目录
    data_dir = os.path.join(base_dir, 'data')
    
    # 检查data文件夹是否存在，如果不存在则创建
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 定义需要创建的子文件夹列表
    sub_dirs = ['分类结果', '需要分类的图片', 'dataset', 'models']
    
    # 遍历子文件夹列表并创建
    for sub_dir in sub_dirs:
        dir_path = os.path.join(data_dir, sub_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# 调用函数
create_data_directories()

def main():
    model_save_path = 'data/models/model.pth'
    data_path = 'data/dataset'
    # 检查模型文件是否存在
    if not os.path.exists(model_save_path):
        print("Model file not found. Starting model training...")

        # 训练模型
        num_epochs = 10
        trained_model, loss_history = train_model(data_path, model_save_path, num_epochs)
        print("Model training completed.")
        print(f"Final loss: {loss_history[-1]}")
    else:
        print("Model file found. Skipping training.")


    # 运行分类
    test_folder = 'data/需要分类的图片'
    output_folder = 'data/分类结果'

    print("Starting image classification...")
    run_classification(data_path, model_save_path, test_folder, output_folder)
    print("Image classification completed.")

if __name__ == "__main__":
    main()