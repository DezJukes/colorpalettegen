from PIL import Image
import numpy as np
import random
import math
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


def rgb_cube_kmeans_palette(img, num_colors, cube_bins=16, count_threshold=1, sample_rate=1.0, max_iter=10):
    """
    Implements the Fast K-Means Original (Huang 2021) palette extraction.
    Returns: (palette_rgb, mse)
    """
    def rgb_to_cube_index(r, g, b, bins):
        rb = (r * bins) // 256
        gb = (g * bins) // 256
        bb = (b * bins) // 256
        return (rb, gb, bb)

    def squared_euclidean(c1, c2):
        dr = c1[0] - c2[0]
        dg = c1[1] - c2[1]
        db = c1[2] - c2[2]
        return dr*dr + dg*dg + db*db

    def build_rgb_cubes(img, bins, count_threshold):
        width, height = img.size
        cube_stats = {}
        for y in range(height):
            for x in range(width):
                r, g, b = img.getpixel((x, y))
                cube_idx = rgb_to_cube_index(r, g, b, bins)
                if cube_idx not in cube_stats:
                    cube_stats[cube_idx] = {"count": 0, "sum_r": 0, "sum_g": 0, "sum_b": 0}
                cube_stats[cube_idx]["count"] += 1
                cube_stats[cube_idx]["sum_r"] += r
                cube_stats[cube_idx]["sum_g"] += g
                cube_stats[cube_idx]["sum_b"] += b
        initc, initn = [], []
        for stats in cube_stats.values():
            if stats["count"] >= count_threshold:
                c_count = stats["count"]
                mean_r = stats["sum_r"] // c_count
                mean_g = stats["sum_g"] // c_count
                mean_b = stats["sum_b"] // c_count
                initc.append((mean_r, mean_g, mean_b))
                initn.append(c_count)
        return initc, initn

    def block_sample_pixels(img, sampling_rate):
        width, height = img.size
        sampled = []
        if abs(sampling_rate - 1.0) < 1e-9:
            for y in range(height):
                for x in range(width):
                    sampled.append(img.getpixel((x, y)))
            return sampled
        # fallback: random sampling
        for y in range(height):
            for x in range(width):
                if random.random() < sampling_rate:
                    sampled.append(img.getpixel((x, y)))
        return sampled

    def initial_palette_generation(initc, initn, K):
        N = len(initc)
        if N == 0:
            return []
        K = min(K, N)
        selected = [False] * N
        palette = []
        Cno = 0
        j = max(range(N), key=lambda i: initn[i])
        selected[j] = True
        palette.append(initc[j])
        Cno += 1
        while Cno < K:
            best_idx = None
            best_score = -1.0
            for i in range(N):
                if selected[i]:
                    continue
                dist_i = min(squared_euclidean(initc[i], p) for p in palette)
                score = dist_i * math.sqrt(initn[i])
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is None:
                break
            selected[best_idx] = True
            palette.append(initc[best_idx])
            Cno += 1
        return palette

    def wu_lin_nearest_color(r, g, b, palette, norms, len2s):
        x2 = r*r + g*g + b*b
        x_norm = math.sqrt(x2)
        K = len(palette)
        lo, hi = 0, K - 1
        best_k = 0
        best_diff = float('inf')
        while lo <= hi:
            mid = (lo + hi) // 2
            diff = norms[mid] - x_norm
            abs_diff = abs(diff)
            if abs_diff < best_diff:
                best_diff = abs_diff
                best_k = mid
            if diff < 0:
                lo = mid + 1
            elif diff > 0:
                hi = mid - 1
            else:
                break
        def consider(idx, sed1_min, nearest_idx):
            y = palette[idx]
            len2 = len2s[idx]
            dot_xy = r*y[0] + g*y[1] + b*y[2]
            sed1 = len2 - 2.0 * dot_xy
            if sed1 < sed1_min:
                return sed1, idx
            return sed1_min, nearest_idx
        sed1_min = float('inf')
        nearest_idx = best_k
        sed1_min, nearest_idx = consider(best_k, sed1_min, nearest_idx)
        i = best_k - 1
        while i >= 0:
            y_norm = norms[i]
            lower_bound = y_norm * (y_norm - 2.0 * x_norm)
            if lower_bound >= sed1_min:
                break
            sed1_min, nearest_idx = consider(i, sed1_min, nearest_idx)
            i -= 1
        i = best_k + 1
        while i < K:
            y_norm = norms[i]
            lower_bound = y_norm * (y_norm - 2.0 * x_norm)
            if lower_bound >= sed1_min:
                break
            sed1_min, nearest_idx = consider(i, sed1_min, nearest_idx)
            i += 1
        sed = x2 + sed1_min
        return nearest_idx, sed

    def fast_kmeans_palette_refinement(sampled_pixels, initial_palette, max_iter=10):
        if not sampled_pixels:
            return initial_palette, 0.0
        palette = [list(c) for c in initial_palette]
        K = len(palette)
        SPN = len(sampled_pixels)
        Iter = 0
        StopF = 0
        prev_mse = None
        while Iter < max_iter and not StopF:
            len2s = []
            norms = []
            for c in palette:
                l2 = c[0]*c[0] + c[1]*c[1] + c[2]*c[2]
                len2s.append(l2)
                norms.append(math.sqrt(l2))
            combined = list(zip(palette, norms, len2s))
            combined.sort(key=lambda t: t[1])
            palette, norms, len2s = zip(*combined)
            palette = [list(c) for c in palette]
            norms = list(norms)
            len2s = list(len2s)
            clusters = [[] for _ in range(K)]
            mse_accum = 0.0
            for (r, g, b) in sampled_pixels:
                k_idx, sed = wu_lin_nearest_color(r, g, b, palette, norms, len2s)
                clusters[k_idx].append((r, g, b))
                mse_accum += sed
            MSE1_iter = mse_accum / SPN
            if Iter > 0 and MSE1_iter >= prev_mse:
                StopF = 1
            prev_mse = MSE1_iter
            for k in range(K):
                if clusters[k]:
                    sr = sum(p[0] for p in clusters[k]) / len(clusters[k])
                    sg = sum(p[1] for p in clusters[k]) / len(clusters[k])
                    sb = sum(p[2] for p in clusters[k]) / len(clusters[k])
                    palette[k] = [int(sr), int(sg), int(sb)]
            palette.sort(key=lambda c: c[0]**2 + c[1]**2 + c[2]**2)
            Iter += 1
        refined_palette = [tuple(c) for c in palette]
        return refined_palette, float(round(prev_mse, 3)) if prev_mse else 0.0

    # --- Main pipeline ---
    img = img.convert("RGB")
    initc, initn = build_rgb_cubes(img, bins=cube_bins, count_threshold=count_threshold)
    initial_palette = initial_palette_generation(initc, initn, K=num_colors)
    sampled_pixels = block_sample_pixels(img, sample_rate)
    final_palette, mse = fast_kmeans_palette_refinement(sampled_pixels, initial_palette, max_iter=max_iter)
    return final_palette, mse