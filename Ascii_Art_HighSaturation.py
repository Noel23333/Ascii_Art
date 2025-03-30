import cv2
import numpy as np
from PIL import Image
import os

# 盲文字符映射
BRAILLE_UNICODE_START = 0x2800
BRAILLE_DOTS = [0b00000001, 0b00000010, 0b00000100, 0b00001000,
                0b00010000, 0b00100000, 0b01000000, 0b10000000]

def rgb_to_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    return lab

def adjust_contrast_brightness(image, contrast=1.0, brightness=0):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def image_to_braille(image_path, output_folder, width=100, contrast=1.0, brightness=0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image = cv2.imread(image_path)
    image = adjust_contrast_brightness(image, contrast, brightness)
    lab_image = rgb_to_lab(image)
    l_channel, a, b = cv2.split(lab_image)
    
    height = int((image.shape[0] / image.shape[1]) * width * 0.5)
    resized = cv2.resize(l_channel, (width * 2, height * 4), interpolation=cv2.INTER_LINEAR)
    
    braille_art = ""
    for y in range(0, resized.shape[0], 4):
        for x in range(0, resized.shape[1], 2):
            braille_char = 0
            for dy in range(4):
                for dx in range(2):
                    if y + dy < resized.shape[0] and x + dx < resized.shape[1]:
                        if resized[y + dy, x + dx] < 128:  # 亮度阈值
                            braille_char |= BRAILLE_DOTS[dy * 2 + dx]
            braille_art += chr(BRAILLE_UNICODE_START + braille_char)
        braille_art += '\n'
    
    output_path = os.path.join(output_folder, "ascii_output.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(braille_art)
    
    print(f"字符画已保存至 {output_path}")
    return braille_art

# 示例使用
ascii_art = image_to_braille("mansui.jpg", "output_folder", width=800, contrast=1.2, brightness=10)
