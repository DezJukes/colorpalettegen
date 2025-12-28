from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb

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
        clusters = [[] for _ in range(K)]
        mse = 0.0

        for p in pixels_lab:
            dists = np.sum((centroids - p) ** 2, axis=1)
            idx = int(np.argmin(dists))
            clusters[idx].append(p)
            mse += dists[idx]

        mse /= N
        mse_history.append(mse)

        for k in range(K):
            if clusters[k]:
                centroids[k] = np.mean(clusters[k], axis=0)

    rgb_centroids = lab2rgb(centroids.reshape(-1, 1, 3)).reshape(-1, 3)
    rgb_centroids = np.clip(rgb_centroids * 255, 0, 255).astype(np.uint8)
    palette_rgb = [tuple(c) for c in rgb_centroids]

    return palette_rgb, centroids, mse_history[-1]

def quantize_image_lab(img, centroids_lab, palette_rgb):
    arr_rgb = np.array(img, dtype=np.float32) / 255.0
    lab_img = rgb2lab(arr_rgb)

    out = np.zeros_like(arr_rgb)
    for y in range(arr_rgb.shape[0]):
        for x in range(arr_rgb.shape[1]):
            dists = np.sum((centroids_lab - lab_img[y, x]) ** 2, axis=1)
            out[y, x] = np.array(palette_rgb[np.argmin(dists)]) / 255.0

    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))
