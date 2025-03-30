#Ascii_Art_Debug.py
import os
import sys
from ctypes import WinDLL
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # 限制 OpenBLAS 线程数
cuda_path = os.environ.get('CUDA_PATH')
if cuda_path:
    nvml_path = os.path.join(cuda_path, "bin", "nvml.dll")
    if os.path.exists(nvml_path):
        os.add_dll_directory(os.path.join(cuda_path, "bin"))
        try:
            WinDLL(nvml_path)
        except Exception as e:
            print(f"⚠️ 预加载NVML失败: {str(e)}")
    else:
        print(f"❌ 未找到nvml.dll于: {nvml_path}")
else:
    print("❌ 未检测到CUDA_PATH环境变量")

import numpy as np
from PIL import Image, ImageEnhance
from skimage.color import rgb2lab
import numba
from numba import cuda
from system_monitor import DynamicSystemMonitor, create_live_progress_bar
from process_monitor import ProcessMonitor, create_process_aware_progress

# Braille Unicode characters ordered by "dot density"
BRAILLE_CHARS_SORTED = [
    '⠀',
    '⠁', '⠂', '⠄', '⠈', '⠐', '⠠', '⡀',
    '⠃', '⠅', '⠆', '⠉', '⠊', '⠌', '⠘', '⠑', '⠒', '⠔', '⠡', '⠤', '⠨', '⠰', '⠢', '⡁', '⡂', '⡄', '⡈', '⡐', '⡠', '⢁', '⢂', '⢄', '⢈', '⢐', '⢠', '⣀', 
    '⠇', '⠋', '⠍', '⠎', '⠓', '⠕', '⠖', '⠙', '⠚', '⠥', '⠦', '⠩', '⠪', '⠬', '⠜',  '⠱', '⠲', '⠴', '⠸', '⠣','⡡', '⡢', '⡤', '⡨', '⡰', '⢡', '⢢', '⢤', '⢨', '⢰', '⢃', '⢅', '⢆', '⢉', '⢊', '⢌', '⢘', '⢑', '⢒', '⢔', '⣁', '⣂', '⣄', '⣈', '⣐', '⣠', 
    '⠞', '⠭', '⠮', '⠏', '⠛', '⠼', '⠧', '⠗', '⠝', '⠺', '⡦', '⡪', '⡬', '⡱', '⡲', '⡴', '⡸', '⡇', '⡋', '⡍', '⡎', '⡓', '⡕', '⡖', '⡙', '⡚', '⡜', '⡥', '⡩', '⢦', '⢪', '⢬', '⢱', '⢲', '⢴', '⢸', '⢥', '⢩', '⣡', '⣢', '⣤', '⣨', '⣰', 
    '⠟', '⠫', '⠻', '⠽', '⠾', '⡭', '⡮', '⡞', '⡧', '⡺', '⡼', '⢫', '⢭', '⢮', '⢧', '⢺', '⢼', '⣦', '⣬', '⣱', '⣲', '⣴', '⣸', '⣇', '⣋', '⣍', '⣎', '⣓', '⣖', '⣙', '⣚', '⣜', '⣥', '⣩', 
    '⠿', '⡯', '⡻', '⡽', '⡾', '⢯', '⢻', '⢽', '⢾', '⣫', '⣭', '⣮', '⣞', '⣧', '⣺', '⣼', '⡫', '⣪', '⣕', 
    '⡿', '⢿', '⣯', '⣻', '⣽', '⣾', 
    '⣿'
]


@numba.jit(nopython=True, fastmath=True, parallel=True)
def rgb_to_lightness_numba(img_array, use_lab):
    if use_lab:
        height, width, _ = img_array.shape
        lightness = np.empty((height, width), dtype=np.float32)  # 显式指定 float32
        
        for y in numba.prange(height):
            for x in numba.prange(width):
                r = img_array[y, x, 0] / 255.0
                g = img_array[y, x, 1] / 255.0
                b = img_array[y, x, 2] / 255.0
                y_val = 0.2126 * r + 0.7152 * g + 0.0722 * b
                lightness[y, x] = 116 * (y_val ** (1/3)) - 16
    else:
        # 确保输出也是 float32
        lightness = (0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]).astype(np.float32)
    
    return lightness
@cuda.jit
def normalize_lightness_kernel(lightness, output, invert):
    y, x = cuda.grid(2)
    if y < lightness.shape[0] and x < lightness.shape[1]:
        val = lightness[y, x]
        if invert:
            output[y, x] = 1.0 - val
        else:
            output[y, x] = val

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def resize_image(image, new_width=100, quality_mode='quality'):
    old_width, old_height = image.size
    aspect_ratio = old_height / old_width
    new_height = int(aspect_ratio * new_width)
    
    # 根据模式选择重采样方法
    resample_method = Image.LANCZOS if quality_mode == 'quality' else Image.BILINEAR
    return image.resize((new_width, new_height), resample=resample_method)

def image_to_braille(image_path, width=100, contrast=1.0, brightness=1.0, 
                    use_lab=True, invert=False, quality_mode='balanced', 
                    use_gpu=True):
    """
    CUDA优化版盲文艺术生成器
    
    Args:
        image_path: 输入图像路径
        width: 输出宽度（字符数）
        contrast: 对比度调整 (>1增加, <1减少)
        brightness: 亮度调整 (>1增加, <1减少)
        use_lab: 是否使用Lab颜色空间 (默认True)
        invert: 是否反转亮度 (默认False)
        quality_mode: 'speed'/'balanced'/'quality' (速度/平衡/质量优先)
        use_gpu: 是否使用GPU加速 (默认True)
    
    Returns:
        盲文艺术字符串
    """
    # 加载和预处理图像
    img = Image.open(image_path).convert('RGB')
    img = adjust_brightness(img, brightness)
    img = adjust_contrast(img, contrast)
    img = resize_image(img, width, quality_mode)
    
    img_array = np.array(img, dtype=np.float32)
    
    # 计算亮度
    if quality_mode == 'quality' and not use_gpu:
        # 最高质量模式使用完整Lab转换
        lab = rgb2lab(img_array)
        lightness = lab[:,:,0]
        lightness_normalized = lightness / 100.0
    else:
        # 使用加速版本
        lightness = rgb_to_lightness_numba(img_array, use_lab and quality_mode != 'speed')
        
        # 归一化
        if use_gpu and cuda.is_available():
            # GPU加速归一化
            lightness_normalized = np.empty_like(lightness)
            threadsperblock = (32, 32)
            blockspergrid_x = (lightness.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
            blockspergrid_y = (lightness.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            
            normalize_lightness_kernel[blockspergrid, threadsperblock](
                lightness, lightness_normalized, invert
            )
            lightness_normalized = (lightness_normalized - lightness_normalized.min()) / \
                                 (lightness_normalized.max() - lightness_normalized.min())
        else:
            # CPU归一化
            lightness_normalized = (lightness - lightness.min()) / (lightness.max() - lightness.min())
            if invert:
                lightness_normalized = 1 - lightness_normalized
    
    # 生成盲文
    monitor = ProcessMonitor()#monitor = DynamicSystemMonitor()
    
    sorted_chars = BRAILLE_CHARS_SORTED
    height, width = lightness_normalized.shape
    result = []
    
    # 创建进度条
    total_rows = (height - height % 4) // 4

    with create_live_progress_bar(total_rows, "生成盲文") as pbar:
        # 根据质量模式选择处理方式
        if quality_mode == 'speed':
            # 快速模式 - 使用更简单的采样
            step_y = 4
            step_x = 2
            for y in range(0, height - step_y, step_y):
                row = []
                for x in range(0, width - step_x, step_x):
                    block = lightness_normalized[y:y+step_y, x:x+step_x]
                    avg = np.mean(block)
                    char_index = min(int(avg * (len(sorted_chars)-1)), len(sorted_chars)-1)
                    row.append(sorted_chars[char_index])
                result.append(''.join(row))
                
                '''
                stats = monitor.get_dynamic_stats()
                pbar.set_postfix(stats, refresh=True)
                pbar.update(1)
                '''
                stats = monitor.get_stats()
                display_text = (
                    f"CPU: {stats['cpu']:.1f}% | "
                    f"Mem: {stats['memory']['rss']:.1f}MB"
                )
                if 'gpu' in stats and stats['gpu']:
                    display_text += f" | GPU Mem: {stats['gpu']['used_memory']:.1f}MB"
                
                pbar.set_postfix_str(display_text, refresh=True)
                pbar.update(1)
        else:
            # 质量/平衡模式 - 使用更精确的处理
            for y in range(0, height - height % 4, 4):
                row = []
                for x in range(0, width - width % 2, 2):
                    block = lightness_normalized[y:y+4, x:x+2]
                    if quality_mode == 'quality':
                        # 质量模式使用加权平均
                        weights = np.array([[0.1, 0.15], 
                                        [0.2, 0.25],
                                        [0.2, 0.25],
                                        [0.1, 0.15]])
                        avg = np.sum(block * weights) / np.sum(weights)
                    else:
                        # 平衡模式使用简单平均
                        avg = np.mean(block)
                    char_index = min(int(avg * (len(sorted_chars)-1)), len(sorted_chars)-1)
                    row.append(sorted_chars[char_index])
                result.append(''.join(row))
                
                '''
                stats = monitor.get_dynamic_stats()
                pbar.set_postfix(stats, refresh=True)
                pbar.update(1)
                '''
                stats = monitor.get_stats()
                display_text = (
                    f"CPU: {stats['cpu']:.1f}% | "
                    f"Mem: {stats['memory']['rss']:.1f}MB"
                )
                if 'gpu' in stats and stats['gpu']:
                    display_text += f" | GPU Mem: {stats['gpu']['used_memory']:.1f}MB"
                
                pbar.set_postfix_str(display_text, refresh=True)
                pbar.update(1)
    return '\n'.join(result)

def save_braille_art(braille_art, filename):
    braille_art = braille_art.replace('\r\n', '\n')
    with open('output_folder\\'+filename, 'w', encoding='utf-8', newline='\n') as f:
        f.write(braille_art)
    
    with open(f"output_folder\\{filename}.html", 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
        <html><body>
        <pre style="font-family: 'Courier New', monospace; font-size: 18px; line-height: 1.1">
        {braille_art}
        </pre></body></html>""")


if __name__ == '__main__':
    DynamicSystemMonitor.print_gpu_info()
    
    image_path = 'mansui.jpg'
    
    # 高质量模式 (GPU加速)
    braille_art_quality = image_to_braille(
        image_path, 
        width=400,
        contrast=1.2,
        brightness=3,
        use_lab=True,
        invert=True,
        quality_mode='quality',
        use_gpu=True
    )
    
    # 快速模式 (CPU)
    braille_art_fast = image_to_braille(
        image_path,
        width=400,
        contrast=1.2,
        brightness=3,
        use_lab=False,
        invert=True,
        quality_mode='speed',
        use_gpu=False
    )
    
    #print(braille_art_quality)  # 打印部分结果
    
    save_braille_art(braille_art_quality, "output_quality.txt")
    save_braille_art(braille_art_fast, "output_fast.txt")