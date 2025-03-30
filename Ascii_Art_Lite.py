import numpy as np
from PIL import Image, ImageEnhance
from skimage.color import rgb2lab

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

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def resize_image(image, new_width=100):
    old_width, old_height = image.size
    aspect_ratio = old_height / old_width
    new_height = int(aspect_ratio * new_width )
    return image.resize((new_width, new_height))

def image_to_braille(image_path, width=100, contrast=1.0, brightness=1.0, use_lab=True, invert=False):
    """
    最终优化版：按视觉密度排序的盲文生成
    支持亮度反转的盲文艺术生成
    
    Args:
        image_path: 输入图像路径
        width: 输出宽度（字符数）
        contrast: 对比度调整 (>1增加, <1减少)
        brightness: 亮度调整 (>1增加, <1减少)
        use_lab: 是否使用Lab颜色空间 (默认True)
        invert: 是否反转亮度 (默认False)
    
    Returns:
        盲文艺术字符串
    """
    img = Image.open(image_path).convert('RGB')
    img = adjust_brightness(img, brightness)
    img = adjust_contrast(img, contrast)
    img = resize_image(img, width)
    
    img_array = np.array(img)
    
    if use_lab:
        lab = rgb2lab(img_array)
        lightness = lab[:,:,0]
        lightness_normalized = lightness / 100.0
    else:
        lightness = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        lightness_normalized = (lightness - lightness.min()) / (lightness.max() - lightness.min())
    
    # 新增：亮度反转
    if invert:
        lightness_normalized = 1 - lightness_normalized
    
    # 使用严格排序的字符集
    sorted_chars = BRAILLE_CHARS_SORTED
    
    # 映射到256级灰度（更平滑的过渡）
    height, width = lightness_normalized.shape
    result = []
    for y in range(0, height - height % 4, 4):
        row = []
        for x in range(0, width - width % 2, 2):
            block = lightness_normalized[y:y+4, x:x+2]
            avg = np.mean(block)
            char_index = min(int(avg * (len(sorted_chars)-1)), len(sorted_chars)-1)
            row.append(sorted_chars[char_index])
        result.append(''.join(row))
    
    return '\n'.join(result)

def save_braille_art(braille_art, filename):
    # 标准化换行符 + 强制UTF-8 + 字体提示
    braille_art = braille_art.replace('\r\n', '\n')
    with open('output_folder\\'+filename, 'w', encoding='utf-8', newline='\n') as f:
        f.write("=== 请用等宽字体查看(if you choose to use ' '(U+0020) instead of '⠀'(U+2800)) ===\n")
        f.write(braille_art)
    
    # 同时生成HTML版本确保渲染一致
    with open(f"output_folder\\{filename}.html", 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
        <html><body>
        <pre style="font-family: 'Courier New', monospace; font-size: 18px; line-height: 1.1">
        {braille_art}
        </pre></body></html>""")

if __name__ == '__main__':
    # Enter your image path here
    image_path = 'reluctant.png'
    
    # e.g. generate braille art with specific parameters
    braille_art = image_to_braille(
        image_path, 
        width=160,     # output width in characters
        contrast=5,   # contrast adjustment factor
        brightness=3, # brightness adjustment factor
        use_lab=True,   # use Lab color space
        invert=True     # turn on brightness inversion
    )
    
    print(braille_art)
    
    save_braille_art(braille_art, "output.txt")