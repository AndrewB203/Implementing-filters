from skimage import filters, io
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
img = io.imread(r"")

# Гауссово размытие
blurred_skimage = filters.gaussian(img, sigma=1.5)

# Усиление резкости 
sharpened_skimage = filters.unsharp_mask(img, amount=1.5, radius=1)

# Визуализация
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img)
axes[0].set_title('Оригинал')
axes[0].axis('off')

axes[1].imshow(blurred_skimage)
axes[1].set_title('Gaussian Blur (skimage)')
axes[1].axis('off')

axes[2].imshow(sharpened_skimage)
axes[2].set_title('Unsharp Mask (skimage)')
axes[2].axis('off')

plt.show()