from scipy.ndimage import convolve
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Загрузка изображения
image_path = ""  # Укажите путь к вашему файлу
# Загружаем изображение с поддержкой Unicode-путей
pil_img = Image.open(image_path).convert('RGB')
# Конвертируем в формат OpenCV (BGR)
img_rgb = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
img = cv2.imread(image_path)

# Ядро для усиления резкости
kernel_sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# Применяем свёртку к каждому каналу
sharpened_scipy = np.zeros_like(img_rgb)
for channel in range(3):
    sharpened_scipy[:, :, channel] = convolve(img_rgb[:, :, channel], kernel_sharpen, mode='reflect')

sharpened_scipy = np.clip(sharpened_scipy, 0, 255).astype(np.uint8)

# Визуализация
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Оригинал')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sharpened_scipy)
plt.title('Sharpening (scipy.ndimage.convolve)')
plt.axis('off')

plt.show()