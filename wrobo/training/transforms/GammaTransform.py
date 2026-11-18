from PIL import Image
import numpy as np

class GammaTransform:
    """Random gamma correction for lighting augmentation"""
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range
    
    def __call__(self, img):
        # img is PIL Image
        gamma = np.random.uniform(*self.gamma_range)
        # Convert to numpy, apply gamma, convert back
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.power(img_array, gamma)
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_array)