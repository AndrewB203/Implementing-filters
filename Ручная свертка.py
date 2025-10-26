import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def manual_convolve(image, kernel):
    
    h, w, c = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Добавляем паддинг
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
    output = np.zeros_like(image)

    # Проходим по каждому пикселю
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                region = padded[y:y+kh, x:x+kw, ch]
                output[y, x, ch] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)

# Загрузка изображения
image_path = ""  # Укажите путь к вашему файлу
# Загружаем изображение с поддержкой Unicode-путей
pil_img = Image.open(image_path).convert('RGB')
# Конвертируем в формат OpenCV (BGR)
img_rgb = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
img = cv2.imread(image_path)

# Ядро для размытия (box blur)
kernel_box = np.ones((5, 5)) / 25

# Применяем ручную свёртку
blurred_manual = manual_convolve(img_rgb, kernel_box)

# Визуализация
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Оригинал')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(blurred_manual)
plt.title('Ручная свёртка (Box Blur)')
plt.axis('off')

plt.show()