# main.py

from src.train_model import train_model
from src.classify_model import run_classification
from src.remove_duplicates import remove_duplicates  # 导入去重模块
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
    if os.path.exists(model_save_path):
        retrain_model = input("模型文件已存在，您想要重新训练模型吗？输入 1 表示是，输入 0 表示否:")
        if retrain_model == '1':
            print("开始重新训练模型...")
            # 重新训练模型
            num_epochs = 20
            trained_model, loss_history = train_model(data_path, model_save_path, num_epochs)
            print("模型重新训练完成。")
            print(f"最终损失: {loss_history[-1]}")
        else:
            print("Using existing model.")
    else:
        print("模型文件未找到，开始训练模型...")
        # 训练模型
        num_epochs = 20
        trained_model, loss_history = train_model(data_path, model_save_path, num_epochs)
        print("模型训练完成。")
        print(f"最终损失: {loss_history[-1]}")


    # 运行分类
    test_folder = 'data/需要分类的图片'
    output_folder = 'data/分类结果'

    print("开始图片分类...")
    run_classification(data_path, model_save_path, test_folder, output_folder)
    print("图片分类完成。")
    
    # 调用去重函数
    print("开始去重...")
    remove_duplicates(output_folder)
    print("去重完成。")
    
    # 新增功能：询问是否清空文件夹
    clear_folder = input("是否清空'data/需要分类的图片'文件夹? 输入 1 表示是，输入 0 表示否: ")
    if clear_folder == '1':
        test_folder = 'data/需要分类的图片'
        for filename in os.listdir(test_folder):
            file_path = os.path.join(test_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print("'data/需要分类的图片' has been cleared.")
    else:
        print("'data/需要分类的图片' has not been cleared.")


if __name__ == "__main__":
    main()