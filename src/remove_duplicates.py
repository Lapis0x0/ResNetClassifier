import os
import imagehash
from PIL import Image

def calculate_perceptual_hash(image_path):
    """计算图片的感知哈希值"""
    with Image.open(image_path) as img:
        return imagehash.phash(img)

def remove_duplicates(output_folder):
    """移除输出文件夹中的重复图片"""
    seen_hashes = set()
    for root, _, files in os.walk(output_folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                file_hash = calculate_perceptual_hash(file_path)
                if file_hash in seen_hashes:
                    os.remove(file_path)
                    print(f"Removed duplicate image: {file_path}")
                else:
                    seen_hashes.add(file_hash)