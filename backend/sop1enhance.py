from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb

def resize_image(img, max_size=512):
    w, h = img.size
    if w > max_size or h > max_size:
        scale = min(max_size / w, max_size / h)
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.LANCZOS)
    return img

def rgb_list_to_lab_array(pixels):
    arr = np.array(pixels, dtype=np.float32) / 255.0
    return rgb2lab(arr.reshape(-1, 1, 3)).reshape(-1, 3)

def kmeans_lab(pixels_rgb, K, max_iter=10, seed=1):
    rng = np.random.RandomState(seed)

    pixels_lab = rgb_list_to_lab_array(pixels_rgb)
    N = len(pixels_lab)
    K = min(K, N)

    centroids = pixels_lab[rng.choice(N, K, replace=False)].copy()
    mse_history = []

    for _ in range(max_iter):
        dists = np.sum((pixels_lab[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        mse = np.mean(np.min(dists, axis=1))
        mse_history.append(mse)

        for k in range(K):
            members = pixels_lab[labels == k]
            if len(members) > 0:
                centroids[k] = np.mean(members, axis=0)

    rgb_centroids = lab2rgb(centroids.reshape(-1, 1, 3)).reshape(-1, 3)
    rgb_centroids = np.clip(rgb_centroids * 255, 0, 255).astype(np.uint8)
    palette_rgb = [tuple(c) for c in rgb_centroids]

    return palette_rgb, centroids, mse_history[-1]

def quantize_image_lab(img, centroids_lab, palette_rgb):
    arr_rgb = np.array(img, dtype=np.float32) / 255.0
    lab_img = rgb2lab(arr_rgb)

    h, w, _ = lab_img.shape
    flat_lab = lab_img.reshape(-1, 3)
    dists = np.sum((flat_lab[:, None, :] - centroids_lab[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    palette_arr = np.array(palette_rgb, dtype=np.float32) / 255.0
    quantized_flat = palette_arr[labels]
    quantized_img = quantized_flat.reshape(h, w, 3)

    return Image.fromarray(np.clip(quantized_img * 255, 0, 255).astype(np.uint8))