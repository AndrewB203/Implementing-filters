import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Загрузка изображения
image_path = ""  # Укажите путь к вашему файлу
# Загружаем изображение с поддержкой Unicode-путей
pil_img = Image.open(image_path).convert('RGB')
# Конвертируем в формат OpenCV (BGR)
img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
img = cv2.imread(image_path)

if img is None:
    print("Ошибка: изображение не найдено!")
    exit()

# Конвертация в RGB (OpenCV загружает в BGR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1. Размытие (Box Blur) — усреднение по окрестности 5x5
kernel_box = np.ones((5, 5), np.float32) / 25
blurred_box = cv2.filter2D(img_rgb, -1, kernel_box)

# 2. Гауссово размытие (более естественное)
blurred_gauss = cv2.GaussianBlur(img_rgb, (5, 5), 0)

# 3. Усиление резкости (Sharpness) — ядро для усиления контуров
kernel_sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
sharpened = cv2.filter2D(img_rgb, -1, kernel_sharpen)

# 4. Выделение границ (Laplacian) — выявляет области с высоким градиентом
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Лапласиан работает лучше на сером
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian_abs = np.uint8(np.absolute(laplacian))  # Приведение к 8-битному изображению
laplacian_rgb = cv2.cvtColor(laplacian_abs, cv2.COLOR_GRAY2RGB)  # Для отображения

# Визуализация всех результатов
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Оригинал')
axes[0, 0].axis('off')

axes[0, 1].imshow(blurred_box)
axes[0, 1].set_title('Box Blur (5x5)')
axes[0, 1].axis('off')

axes[1, 0].imshow(blurred_gauss)
axes[1, 0].set_title('Gaussian Blur (5x5)')
axes[1, 0].axis('off')

axes[1, 1].imshow(sharpened)
axes[1, 1].set_title('Sharpening Filter')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# Отдельно показываем Laplacian (он в ч/б, но конвертировали в RGB)
plt.figure(figsize=(6, 6))
plt.imshow(laplacian_rgb)
plt.title('Laplacian Edge Detection')
plt.axis('off')
plt.show()