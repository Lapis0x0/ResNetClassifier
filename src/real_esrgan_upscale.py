import torch
from realesrgan import RealESRGAN
from PIL import Image
import os

def upscale_image(input_path, output_path, scale=4):
    # 加载Real-ESRGAN模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RealESRGAN(device, scale=scale)
    model.load_weights('weights/RealESRGAN_x4.pth')

    # 打开输入图片
    image = Image.open(input_path).convert('RGB')

    # 放大图片
    sr_image = model.predict(image)

    # 保存放大后的图片
    sr_image.save(output_path)
    print(f"Image saved to {output_path}")

def upscale_images_in_folder(input_folder, output_folder, scale=4):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    total_files = len(files)
    for index, filename in enumerate(files, start=1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        upscale_image(input_path, output_path, scale)
        print(f"Processed {index}/{total_files}: {filename}")
        
if __name__ == "__main__":
    # 示例用法
    input_folder = 'data/需要分类的图片'
    output_folder = 'data/分类结果/放大图片'
    upscale_images_in_folder(input_folder, output_folder)