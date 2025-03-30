# 盲文画生成算法

> 当现实中失去了寄托，也许2D是唯一的归宿……

## 文件结构

```basic
Ascii_Art/
│
├── __pycache__/
│   ├── process_monitor.cpython-39.pyc
│   └── system_monitor.cpython-39.pyc
│
├── output_folder/
│   ├── output.txt
│   ├── output.txt.html
│   ├── output_fast.txt
│   ├── output_fast.txt.html
│   ├── output_quality.txt
│   └──  output_quality.txt.html
│
├── Ascii_Art.py
├── ASCII_Art_Debug.py
├── Ascii_Art_HighSaturation.py
├── Ascii_Art_Lite.py
├── GPU_Check.py
├── GPU_Detect.py
├── GPU_Detect_History.py
├── mansui.jpg
├── process_monitor.py
├── ReadMe.md
└── system_monitor.py
```

## 食用方法

### 环境搭建

#### 关键库与个别版本要求

* python 3.9
* numpy 1.22.4
* opencv-python
* scikit-image 0.22.0
* numba 0.53.0
* cudatoolkit（此项建议使用conda安装以解决CUDA依赖问题）
* scipy 1.13.1

==Warning：不要轻易尝试在已搭建好的环境中随意更改库版本==

### 速食方法

1. 选择Ascii_Art.py版本，（CPU用户可以选择Lite版本）。

2. 在主程序处更改欲处理的图片所在路径，更改参数以求得想要的效果。

3. 具体参数解释如下：

   ##### 显式参数与功能

   | 参数         | 解释                                                   |
   | ------------ | ------------------------------------------------------ |
   | image_path   | 输入图像路径                                           |
   | width        | 输出宽度（字符数）                                     |
   | contrast     | 对比度调整 (>1增加, <1减少)                            |
   | brightness   | 亮度调整 (>1增加, <1减少)                              |
   | use_lab      | 是否使用Lab颜色空间 (默认True)                         |
   | invert       | 是否反转亮度 (默认False)                               |
   | quality_mode | 'speed'/'balanced'/'quality'<br />(速度/平衡/质量优先) |
   | use_gpu      | 是否使用GPU加速 (默认True)                             |

   ##### 质量模式差异

   | 模式 | 亮度计算 | 分块处理    | 权重       | 适用场景 |
   | :--- | :------- | :---------- | :--------- | :------- |
   | 速度 | RGB加权  | 4x2简单平均 | 无         | 快速预览 |
   | 平衡 | LAB/RGB  | 4x2简单平均 | 无         | 日常使用 |
   | 质量 | LAB      | 4x2加权平均 | 中心权重高 | 精细输出 |

生成的一些输出已经存储在 /output_folder/ 下，可供参考。

*如果你对原理毫无兴趣，那么可以止步于此了~*~

## 原理与个性化参数调校

下面给出代码的详细原理讲解，解释每个部分的功能和设计思路。

这份代码的主要目标是将一张输入图片转换为“盲文画”风格的文本输出。整个过程包括图像预处理、光度计算、归一化、采样以及最终根据光度值选取对应的盲文字符。下面按模块进行详细说明。

### 1. 导入库与环境配置

```python
import numpy as np
from PIL import Image, ImageEnhance
from skimage.color import rgb2lab
import numba
from numba import cuda
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # restrict OpenBLAS thread count
```

- **NumPy**：用于数组操作和数值计算。
- **PIL (Pillow)**：用于图像加载和处理，如调整亮度、对比度、和图像缩放。
- **scikit-image (skimage)**：提供 RGB 到 Lab 颜色空间的转换，主要在“高质量”模式下使用。
- **Numba**：利用 JIT 编译加速 Python 函数的执行，其中 `@numba.jit` 用于加速光度计算；`@cuda.jit` 用于 GPU 加速归一化计算。
- **os 环境变量设置**：通过设置 `OPENBLAS_NUM_THREADS` 限制 Open BLAS 线程数，防止多线程干扰性能。

### 2. 盲文字符集

```python
BRAILLE_CHARS_SORTED = [
    '⠀',
    '⠁', '⠂', '⠄', '⠈', '⠐', '⠠', '⡀',
    ...
    '⣿'
]
```

- **作用**：定义一个排序好的盲文 Unicode 字符列表，字符按照“点密度”排序。生成最终盲文画时，会根据图像的光度值映射到对应的字符上。
- **原理**：图像的光度值经过归一化后映射为区间 [0, 1]，乘以字符列表的长度，得到一个整数索引，从而选出合适的盲文字符表示图像的局部亮度。

### 3. 光度计算：`rgb_to_lightness_numba`

```python
@numba.jit(nopython=True, fastmath=True, parallel=True)
def rgb_to_lightness_numba(img_array, use_lab):
    if use_lab:
        height, width, _ = img_array.shape
        lightness = np.empty((height, width), dtype=np.float32)
        
        for y in numba.prange(height):
            for x in numba.prange(width):
                r = img_array[y, x, 0] / 255.0
                g = img_array[y, x, 1] / 255.0
                b = img_array[y, x, 2] / 255.0
                y_val = 0.2126 * r + 0.7152 * g + 0.0722 * b
                lightness[y, x] = 116 * (y_val ** (1/3)) - 16
    else:
        lightness = (0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]).astype(np.float32)
    
    return lightness
```

- **功能**：将 RGB 图像转换为光度（亮度）图。

- **两种模式**：

  - **使用 Lab 颜色空间**

    1. **归一化**：每个通道除以 255 得到范围 $[0,1]$ 内的值。

    2. **加权求和**：使用公式
       $y\_val = 0.2126 \times r + 0.7152 \times g + 0.0722 \times b$

    3. **伽马校正**：进行立方根运算并转换为 Lab 空间中的 L 值
       $L = 116 \times \sqrt[3]{y\_val} - 16$

  - **直接加权平均**：使用常见的加权系数 (0.299, 0.587, 0.114) 进行线性加权平均计算亮度：
        $L = 0.299 \times R + 0.587 \times G + 0.114 \times B$

- **加速方式**：使用 Numba 的 `@jit` 加速，启用 `fastmath` 和并行计算 (`numba.prange`)，提高大图像处理时的性能。

### 4. GPU 归一化处理：`normalize_lightness_kernel`

```python
@cuda.jit
def normalize_lightness_kernel(lightness, output, invert):
    y, x = cuda.grid(2)
    if y < lightness.shape[0] and x < lightness.shape[1]:
        val = lightness[y, x]
        if invert:
            output[y, x] = 1.0 - val
        else:
            output[y, x] = val
```

- **功能**：对光度矩阵进行归一化处理，并支持反转亮度（invert）。

- **使用 CUDA**：通过 GPU 平台加速归一化过程。利用 `cuda.grid(2)` 获取当前线程处理的二维坐标，确保在矩阵边界内进行计算。

- 若 `invert` 为 True，则将光度值反转，即：
      $\text{output}[y,x] = 1.0 - \text{val}$

  否则保持原值。

### 5. 图像调整与缩放

#### 调整对比度与亮度

```python
def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)
```

- **功能**：利用 PIL 中的增强模块对图像的对比度和亮度进行调整。
- **参数**：
  - `factor > 1`：增强对比度或亮度；`factor < 1`：降低对比度或亮度。

#### 图像缩放

```python
def resize_image(image, new_width=100, quality_mode='quality'):
    old_width, old_height = image.size
    aspect_ratio = old_height / old_width
    new_height = int(aspect_ratio * new_width)
    
    resample_method = Image.LANCZOS if quality_mode == 'quality' else Image.BILINEAR
    return image.resize((new_width, new_height), resample=resample_method)
```

- **功能**：调整图像尺寸，保持原有宽高比例。
- **质量模式**：根据用户选择的质量模式决定使用的重采样方法：
  - **质量模式**：使用 LANCZOS 滤波，适合对质量要求较高的情况。
  - **速度模式**：使用 BILINEAR 滤波，速度更快但质量略低。

### 6. 图像转换为盲文画：`image_to_braille`

```python
def image_to_braille(image_path, width=100, contrast=1.0, brightness=1.0, 
                    use_lab=True, invert=False, quality_mode='balanced', 
                    use_gpu=True):
    ...
```

#### 主要步骤

1. ### 主要步骤

   1. **图像加载与预处理**

      - 使用 PIL 加载图像并转换为 RGB 模式。
      - 调整图像的亮度和对比度。
      - 根据指定的输出宽度和质量模式对图像进行缩放。

   2. **转换为 NumPy 数组**

      - 将预处理后的图像转换为 NumPy 数组，便于后续的数值计算。

   3. **光度计算**

      - **高质量模式**：如果 `quality_mode` 为 `"quality"` 且不使用 GPU，则利用 `rgb2lab` 转换图像，直接获取 Lab 空间中的 L 值，其归一化公式为
        $L_{\text{norm}} = \frac{L}{100}$
      - **其他模式**：调用 `rgb_to_lightness_numba` 计算光度，根据 `use_lab` 参数决定使用 Lab 公式还是线性加权法。

   4. **归一化光度**

      - **GPU 加速归一化**：如果启用了 GPU 且 CUDA 可用，则调用 `normalize_lightness_kernel` 在 GPU 上进行归一化处理。
      - **CPU 归一化**：直接计算
        $L_{\text{normalized}} = \frac{L - L_{\min}}{L_{\max} - L_{\min}}$ 同时根据 `invert` 参数决定是否反转光度（即取 $1 - L_{\text{normalized}}$）。

   5. **生成盲文字符矩阵**

      - 根据采样方法（由质量模式决定）将图像分块：

        - **速度模式**：每块使用较大区域（例如 $4 \times 2$）来快速计算平均亮度。
        - **质量模式**：使用较小的区域并在 `"quality"` 模式下使用加权平均计算细节：
              $\text{avg} = \frac{\sum (w_{ij} \times L_{ij})}{\sum w_{ij}}$

      - 最后将平均亮度映射到预定义的盲文字符列表中，索引计算为
            $\text{index} = \min\left(\left\lfloor \text{avg} \times (N - 1) \right\rfloor, N - 1\right)$ 

        其中 $N$ 为盲文字符数。

   6. **返回结果**

      - 生成一个多行字符串，每行包含对应图像行转换的盲文字符，最终返回整个盲文画字符串。

### 7. 保存盲文画到文件

```python
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
```

- **功能**：
  - 将生成的盲文画字符串保存为纯文本文件。
  - 同时生成一个简单的 HTML 文件，将盲文画嵌入 `<pre>` 标签中，便于在浏览器中预览（防止在不同的操作系统中因字体差异引起的显示异常）。
- **注意**：文件保存路径中使用了 `output_folder\\` 作为输出文件夹。

### 8. 主程序入口

```python
if __name__ == '__main__':
    image_path = 'mansui.jpg'
    
    # Sample use(s GPU and quality mode)
    braille_art_quality = image_to_braille(
        image_path, 
        width=1600,
        contrast=1.2,
        brightness=3,
        use_lab=True,
        invert=True,
        quality_mode='quality',
        use_gpu=True
    )
    
    # Sample use(s CPU and speed mode)
    braille_art_fast = image_to_braille(
        image_path,
        width=1600,
        contrast=1.2,
        brightness=3,
        use_lab=False,
        invert=True,
        quality_mode='speed',
        use_gpu=False
    )
    
    print(braille_art_quality)
    
    save_braille_art(braille_art_quality, "output_quality.txt")
    save_braille_art(braille_art_fast, "output_fast.txt")
```

- **功能**：
  - 指定待处理图像路径。
  - 演示两种不同的调用方式：
    - **高质量模式（使用 GPU 加速）**：`quality` 模式下使用 Lab 空间计算光度，适合对细节和视觉效果要求较高的场景。
    - **快速模式（使用 CPU）**：`speed` 模式下简化计算，适合快速预览或在资源受限情况下使用。
  - 将结果打印到控制台并保存为文本与 HTML 文件，方便后续查看。

### 总结

通过图像预处理、加速计算（Numba 与 CUDA）、以及细致的采样策略，我们实现了从彩色图片生成盲文画图像的功能。

- **灵活性**：通过参数调整（宽度、对比度、亮度、质量模式、是否使用 GPU 等）可以平衡图像效果和计算速度。
- **优化策略**：利用 JIT 编译和 GPU 加速，使得大尺寸图片的处理也能高效执行。
- **输出**：生成的盲文画既可以直接在终端中展示，也可以通过生成 HTML 文件在浏览器中预览。

> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣼⠺⠺⠺⡪⡪⡬⠙⠎⠎⠓⠓⠍⡜⣞⡻⣇⣙⣖⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣽⣼⣾⣽⢯⢯⢯⢻⢻⢻⣓⣍⣍⣍⣍⣋⣫⣺⣽⣞⢯⠿⣧⣿⣿⣿⣾⣾⣿⣿⣿⣿⣿⣿⣿
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⣽⣿⣿⢿⣇⢼⡺⡓⠏⠏⠮⢉⢆⢆⢆⢆⢆⢆⢆⢆⢒⠗⡩⢫⣸⢿⣾⣿⣿⣽⣻⣾⣿⣿⣿⣿⣿⣿⣿
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡫⣪⣾⣙⡎⣭⣿⣧⡺⠛⠦⠰⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⠰⢐⢒⡥⣚⣮⣾⣿⣾⣕⡫⡫⡿⣯⣿⣿⣿
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣺⣪⣾⣽⢿⣤⡰⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠊⢒⣸⣽⣿⣻⣪⣕⢿⣿⣿⣿⣿
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⣻⣓⢊⠌⠂⢂⠓⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠣⣬⣽⣿⣾⣽⣿⣿⣿⣿
> ⣿⣽⣽⣽⣽⣽⣽⣾⣿⣿⣿⣿⣿⢮⡽⣾⣎⡲⠃⠂⠦⠼⠭⠑⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢨⡻⣾⣾⣾⣿⣿⣿
> ⣿⣽⣽⣻⣼⣬⣸⣬⣺⣿⡽⣕⣿⣿⣧⡩⡀⠀⠍⣢⠺⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡁⢭⣾⣿⣿⣿⣿⣿
> ⣿⣾⣻⢿⣕⢮⡮⢫⣪⣿⣫⢿⣾⣕⡤⠀⠐⡲⢦⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠌⢱⢿⣿⣿⣿⣿⣿
> ⣿⣁⠺⣽⣾⣸⡢⡢⡢⡢⡇⣧⣭⢌⠄⠂⣂⢦⠅⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠩⠍⠂⠑⢴⣽⣿⣍⢼⡓
> ⣿⢂⢉⣿⣧⠡⠀⠀⠀⠀⠀⠝⠻⠁⠐⡇⡦⠅⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠉⠀⠀⠀⠄⠀⠀⠀⠀⠀⠀⠁⡄⡐⡄⡈⠺⣕⠣⠀⠐⡧⣾⡫⡱⠇
> ⣿⢐⡮⣿⣧⠆⠀⠀⠀⠀⠀⠨⡭⠀⢤⡼⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠦⢉⠄⠐⠈⠞⢄⠀⠀⠀⠀⠀⠀⡠⣥⡧⡕⠱⠼⣚⠊⠀⡂⣞⣿⡯⠬
> ⣿⠭⢲⣿⣪⠥⠀⠀⠀⠀⠀⡐⢫⠜⣱⡁⠀⠀⠀⠀⠀⠀⠀⠀⠄⣂⣢⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠲⣍⢺⢭⡩⣈⢱⢑⠈⠀⠀⠀⠀⠀⣂⣄⠈⠀⠀⠻⣰⠀⠁⢰⣿⣾⠿
> ⣿⡙⢡⣙⣿⢼⡀⠀⡀⡁⠀⠭⣺⣥⠱⠁⠀⠀⠀⠐⠐⠀⠀⠀⢅⢽⡧⠔⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠖⣓⠖⠀⠀⠀⡂⣬⠙⠀⠀⠀⠀⠀⠄⡞⡁⠀⠀⡁⡼⠈⠀⠨⣦⣿⣻
> ⣿⡧⣰⢱⣕⣾⣰⠑⠑⠼⠸⢒⣻⢽⢈⠀⠀⠀⠠⠰⠁⠀⠀⠲⢯⣻⡮⠔⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠨⠅⠀⠀⠀⠀⠅⠔⠁⠀⠀⠀⠀⠀⡴⡜⠀⠀⠌⣰⠬⠀⠀⣐⣾⣺
> ⣿⣲⣜⣚⣽⣿⢾⣩⣸⠞⠞⢉⣭⠮⠁⠀⠀⠀⡠⡸⠀⠀⠰⣡⣱⠏⡎⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠔⡀⠀⠀⠀⠀⠅⠴⠌⠀⠀⠀⠀⠀⠘⣚⢊⠬⡋⣓⢃⠀⠀⢄⣓
> ⣿⣿⣿⣿⢒⠎⠌⣄⡓⠀⠀⠡⣯⣈⠀⠀⠀⠁⣁⡪⠀⠀⠑⠕⡀⠌⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠑⠈⣀⢄⢁⠊⠇⢉⠡⠀⠀⠀⠀⠀⠄⡻⣼⢴⢑⣴⠭⠀⠀⠀⠽
> ⣿⡿⣿⣿⣪⡸⢐⡭⣇⠀⠀⠥⢯⡨⠀⠀⠀⠄⢴⡎⠀⠀⠌⣀⢁⠚⡂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠈⠣⢄⠡⠠⠔⠃⠨⠀⠀⠀⠀⠀⠄⣓⣰⠈⠃⢥⠏⠀⠀⠂⡮
> ⣿⣴⣼⣿⣦⡋⡸⣙⣚⠢⠅⢫⠫⠀⠀⠀⠀⠁⢅⢠⠀⠂⠤⠒⠃⢐⠢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⡁⠂⠐⡠⠀⠂⡠⠈⠀⠑⢁⠀⠐⡂⠀⠀⠀⠀⠂⡲⡺⠉⠃⢪⣐⠀⠀⡂⣇
> ⣿⣪⣕⣺⡯⢌⡢⣬⡫⣮⣙⣪⠟⠀⠀⠀⠀⠁⡡⠀⠀⠄⢁⠊⠀⠄⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠉⠀⠈⠒⠀⠀⠀⠀⠄⠚⠀⠀⡀⠓⠀⠀⠀⠀⠁⣁⣇⡂⠉⠏⢁⠀⠂⠗⣾
> ⣿⢾⣜⡥⣡⣻⣻⡜⡜⢧⠴⡯⠫⠀⠀⠀⠀⠀⠑⠤⠀⠅⢢⠆⠀⠈⠅⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⢁⠀⠀⡐⠀⠀⠀⠀⠀⠁⠀⠀⠀⡐⡜⠰⠀⠀⠀⠁⢘⣖⡠⠁⠠⠀⠀⢂⠿⣿
> ⣿⠽⡧⣡⡼⣯⡇⡀⢆⢴⠑⣥⣨⠀⠀⠀⠀⠀⢠⢁⠀⠀⠈⠈⠀⠀⠀⠀⠀⠁⠈⠈⠀⠀⠀⠀⠀⠀⠂⠀⠀⠀⠀⢂⢰⢉⢅⠏⡎⠏⣠⠗⠭⡄⠀⠀⠀⠁⡲⢾⠖⠀⠀⠀⠈⡋⣿⣿
> ⣿⡰⠧⣓⣽⣦⠠⠁⡪⢴⡀⣖⣤⠀⠀⠀⠀⠀⠢⣀⠀⠈⢡⣨⡲⢘⡴⡇⡪⢬⢧⡖⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⠋⠪⠪⢐⠠⠈⠂⠁⠘⠃⠀⠀⠀⠂⠻⣪⠱⠀⠀⠘⡕⣻⣿
> ⣿⠞⢥⣼⣾⢭⠀⠀⡐⢢⠀⣇⣤⠀⠀⠀⠀⠀⠆⢡⡄⠀⠀⠄⠒⠤⡂⠨⠤⠰⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⡠⠀⠀⠀⠀⠄⣥⢺⠡⠀⠀⡲⣾⣿⣿⣮⣎
> ⣿⡍⢥⣮⣿⡾⢊⠉⠀⠀⠀⣇⣢⡀⠐⠀⠀⠀⠆⠴⠇⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⢄⠀⠀⠀⠀⢄⢾⢱⠀⠄⠦⠟⣿⣿⣿⣯⣍⣰⣰
> ⣿⢻⢧⢫⡲⡴⡼⡯⢮⡴⠛⢾⡮⢘⢅⠄⠀⠀⠀⡄⡖⡢⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠢⡡⠀⠀⠀⠁⣤⣾⣢⢢⢧⣻⣿⣿⣿⣿⣿⣋
> ⣿⠿⣇⣙⢥⡭⠣⡢⡪⡼⣋⣯⢿⣪⡯⠆⠀⠀⠀⠀⠝⣎⠝⠭⢐⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠣⣇⡰⠀⠀⠁⢉⡫⣾⣾⣻⣽⣾⣿⣿⣿⣿⣿⣋
> ⣿⡯⣚⣚⡾⣮⣲⣍⠫⣙⠫⣞⣯⣿⣽⡙⠀⠀⠀⠀⠂⣢⡿⢬⡡⡸⠛⠪⢈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⢈⠦⠞⢫⠽⢑⠁⠀⠄⠮⢿⣿⢿⣓⣖⢺⣓⠿⠿⠿⠿⠓
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢽⣧⡿⣿⣿⣭⢰⠁⠀⠀⠀⡀⡙⣪⡋⡈⢠⠙⢅⣠⢮⡴⠨⠸⡨⠌⠀⠀⠀⠀⠀⠠⢊⢱⣱⢰⡈⠤⡁⡩⢺⠰⠀⠈⠝⣻⣾⣍⠬⢅⠣⡡⢡⢌⢌⣿
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⣾⣿⣿⣿⣧⣂⠄⠀⠀⠀⠃⢩⢿⣽⣼⣮⣞⣺⣽⡿⠽⣤⣈⢠⢄⡁⠀⠀⠍⡤⠨⠀⠢⡄⣁⢬⣪⣋⢂⠀⠢⠽⣯⣿⣿⣋⠿⣫⢼⣦⣲⣋⣋⣿
> ⣿⣿⣿⣿⣿⣿⣿⡽⣯⢽⣚⣚⣚⣚⣚⣚⣚⣎⡭⣰⢼⠿⣂⢈⣨⣽⣿⣿⣿⣾⣾⣯⡧⠬⠅⢨⠨⠄⠀⡠⠋⠂⠀⠀⠀⠓⡧⣾⣿⢧⠤⠃⡖⣾⣽⣮⣢⠞⠞⠞⠞⠞⠞⠞⣠
> ⣿⣿⣿⣿⣿⣿⣿⡕⢯⢌⠂⠀⠀⠀⠀⠀⠀⠀⠀⠌⠮⢾⣾⣾⣾⣾⣿⣿⣿⣿⣿⠿⠐⠀⠀⠆⠎⠪⠠⠀⠀⠀⠀⠊⢤⢱⣠⠦⢃⣚⢈⠕⣞⣾⢬⡁⠀⠀⠀⠀⠀⠀⠀⠀⠠⠞⠿⣿
> ⣿⣿⣿⣿⣿⣿⣿⡋⡙⣽⣬⡍⠓⠌⠀⠀⠀⠀⠀⠀⠀⠃⣠⣦⣱⣽⣿⣾⣿⣾⣽⣻⣩⢒⡀⠀⠀⠜⡬⠒⠀⠀⠀⠁⠄⠀⠀⠄⢫⡙⠀⠨⡡⠱⠑⠀⠀⠀⠀⠀⠀⢈⡨⡩⡽⣼⡚⣿
> ⣿⣿⣿⣿⣿⣿⣿⢡⠡⢨⡚⢴⣽⢾⣤⢤⠀⠀⠀⠀⠀⠀⠀⠀⠀⡁⡈⡈⠝⡋⡋⡬⠇⠛⣚⣎⢧⣍⠟⣦⠞⢺⡼⠽⣬⢰⠀⠂⡜⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡴⡿⢧⡚⠸⢐⠘⣿
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢦⣋⣫⢒⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢃⣴⢧⡲⢱⠽⠾⡭⡭⢮⣎⣕⡲⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠒⡼⣿⢪⠺⣿⣿⣿⣿⣿
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⢤⠿⣻⢩⡂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠐⠁⠀⠀⠀⠀⠁⠌⠘⠌⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠆⡡⣴⣿⣿⢻⣽⣿⣿⣿
> ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣠⠀⠎⣾⣾⢭⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄⣈⣿⣿⣿⣿⣿⣿⣿⣿
