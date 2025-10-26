from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
# Загрузка изображения
image_path = r""  # Укажите путь к вашему файлу
# Загружаем изображение с поддержкой Unicode-путей
img_pil = Image.open(image_path).convert('RGB')

# Ядро для размытия (box blur)
kernel_data = [1]*25  # 5x5
kernel = ImageFilter.Kernel(
    size=(5, 5),
    kernel=kernel_data,
    scale=25,  # деление на 25
    offset=0
)

blurred_pil = img_pil.filter(kernel)

# Конвертация в numpy для отображения
blurred_np = np.array(blurred_pil)

# Визуализация
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.array(img_pil))
plt.title('Оригинал')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(blurred_np)
plt.title('Box Blur (PIL)')
plt.axis('off')

plt.show()