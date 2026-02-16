from PIL import Image
import numpy as np
import random
import math
from skimage.color import rgb2lab, lab2rgb

# --- PARAMETERS (used in function defaults) ---
PROCESS_MAX = 512
DELTA_E_THRESH = 4.0
SOP3_PASSES = 2
SOP3_CANDIDATES = 2500
QUANT_PREVIEW_MAX = 256

def resize_image(img, max_size=512):
    w, h = img.size
    if w > max_size or h > max_size:
        scale = min(max_size / w, max_size / h)
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.LANCZOS)
    return img

def rgb_list_to_lab_array(rgb_list):
    arr = np.asarray(rgb_list, dtype=np.float32) / 255.0
    lab = rgb2lab(arr.reshape(-1, 1, 3)).reshape(-1, 3)
    return lab.astype(np.float32)

def lab_array_to_rgb_list(lab_arr):
    rgb = lab2rgb(lab_arr.reshape(-1, 1, 3)).reshape(-1, 3)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return [tuple(map(int, c)) for c in rgb_u8]

def rgb_to_cube_index(r, g, b, bins):
    return ((r * bins) // 256, (g * bins) // 256, (b * bins) // 256)

def squared_euclidean_rgb(c1, c2):
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr*dr + dg*dg + db*db

def build_rgb_cubes(img, bins, count_threshold):
    W, H = img.size
    cube_stats = {}
    for y in range(H):
        for x in range(W):
            r, g, b = img.getpixel((x, y))
            idx = rgb_to_cube_index(r, g, b, bins)
            if idx not in cube_stats:
                cube_stats[idx] = [0, 0, 0, 0]
            cube_stats[idx][0] += 1
            cube_stats[idx][1] += r
            cube_stats[idx][2] += g
            cube_stats[idx][3] += b
    initc, initn = [], []
    for count, sr, sg, sb in cube_stats.values():
        if count >= count_threshold:
            initc.append((sr // count, sg // count, sb // count))
            initn.append(count)
    return initc, initn

def initial_palette_generation(initc, initn, K):
    N = len(initc)
    if N == 0:
        return []
    K = min(K, N)
    selected = [False] * N
    palette = []
    j = max(range(N), key=lambda i: initn[i])
    selected[j] = True
    palette.append(initc[j])
    while len(palette) < K:
        best_i, best_score = None, -1.0
        for i in range(N):
            if selected[i]:
                continue
            dist_i = min(squared_euclidean_rgb(initc[i], p) for p in palette)
            score = dist_i * math.sqrt(initn[i])
            if score > best_score:
                best_score, best_i = score, i
        if best_i is None:
            break
        selected[best_i] = True
        palette.append(initc[best_i])
    return palette

def block_sample_pixels(img, sampling_rate):
    W, H = img.size
    sampled = []
    if abs(sampling_rate - 1.0) < 1e-9:
        sampled = list(img.getdata())
        return sampled
    if abs(sampling_rate - 0.5) < 1e-9:
        bs, spb = 2, 2
    elif abs(sampling_rate - 0.25) < 1e-9:
        bs, spb = 2, 1
    elif abs(sampling_rate - 0.125) < 1e-9:
        bs, spb = 4, 2
    elif abs(sampling_rate - 0.0625) < 1e-9:
        bs, spb = 4, 1
    elif abs(sampling_rate - 0.03125) < 1e-9:
        bs, spb = 8, 2
    else:
        for y in range(H):
            for x in range(W):
                if random.random() < sampling_rate:
                    sampled.append(img.getpixel((x, y)))
        return sampled
    for by in range(0, H, bs):
        for bx in range(0, W, bs):
            coords = [(x, y)
                      for y in range(by, min(by + bs, H))
                      for x in range(bx, min(bx + bs, W))]
            if not coords:
                continue
            chosen = random.sample(coords, min(spb, len(coords)))
            for (x, y) in chosen:
                sampled.append(img.getpixel((x, y)))
    return sampled

def _prep_wulin_3d(palette_3d):
    pal = np.asarray(palette_3d, dtype=np.float32)
    len2s = np.sum(pal * pal, axis=1)
    norms = np.sqrt(len2s)
    order = np.argsort(norms)
    return pal[order], norms[order], len2s[order], order

def wu_lin_nearest_color_3d(x0, x1, x2, pal_sorted, norms, len2s):
    x_sq = x0*x0 + x1*x1 + x2*x2
    x_norm = math.sqrt(x_sq)
    Kp = pal_sorted.shape[0]
    lo, hi = 0, Kp - 1
    best_k = 0
    best_diff = float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        diff = norms[mid] - x_norm
        ad = abs(diff)
        if ad < best_diff:
            best_diff = ad
            best_k = mid
        if diff < 0:
            lo = mid + 1
        elif diff > 0:
            hi = mid - 1
        else:
            break
    def consider(idx, sed1_min, nearest_idx):
        y = pal_sorted[idx]
        len2 = float(len2s[idx])
        dot_xy = x0*y[0] + x1*y[1] + x2*y[2]
        sed1 = len2 - 2.0 * dot_xy
        if sed1 < sed1_min:
            return sed1, idx
        return sed1_min, nearest_idx
    sed1_min = float("inf")
    nearest_idx = best_k
    sed1_min, nearest_idx = consider(best_k, sed1_min, nearest_idx)
    i = best_k - 1
    while i >= 0:
        y_norm = float(norms[i])
        if y_norm * (y_norm - 2.0 * x_norm) >= sed1_min:
            break
        sed1_min, nearest_idx = consider(i, sed1_min, nearest_idx)
        i -= 1
    i = best_k + 1
    while i < Kp:
        y_norm = float(norms[i])
        if y_norm * (y_norm - 2.0 * x_norm) >= sed1_min:
            break
        sed1_min, nearest_idx = consider(i, sed1_min, nearest_idx)
        i += 1
    return nearest_idx, (x_sq + sed1_min)

def fast_kmeans_refine_lab(sampled_rgb, initial_palette_rgb, max_iter=10):
    if not sampled_rgb:
        return initial_palette_rgb, [], None, None
    K = len(initial_palette_rgb)
    sampled_lab = rgb_list_to_lab_array(sampled_rgb)
    palette_lab = rgb_list_to_lab_array(initial_palette_rgb)
    mse_hist = []
    prev_mse = None
    last_counts = None
    for it in range(max_iter):
        # Vectorized assignment step
        # Compute squared distances between all sampled_lab and palette_lab
        diff = sampled_lab[:, None, :] - palette_lab[None, :, :]
        dists = np.sum(diff ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        # Update palette
        counts = np.zeros(K, dtype=np.int32)
        sums = np.zeros((K, 3), dtype=np.float64)
        for k in range(K):
            mask = (labels == k)
            counts[k] = np.sum(mask)
            if counts[k] > 0:
                sums[k] = np.sum(sampled_lab[mask], axis=0)
        mse = np.mean(np.min(dists, axis=1))
        mse_hist.append(mse)
        if prev_mse is not None and mse >= prev_mse:
            break
        prev_mse = mse
        for k in range(K):
            if counts[k] > 0:
                palette_lab[k] = (sums[k] / counts[k]).astype(np.float32)
        lens = np.sum(palette_lab * palette_lab, axis=1)
        palette_lab = palette_lab[np.argsort(lens)]
        last_counts = counts.copy()
    final_palette_rgb = lab_array_to_rgb_list(palette_lab)
    return final_palette_rgb, mse_hist, palette_lab, last_counts

def delta_e76_matrix(lab_arr):
    diff = lab_arr[:, None, :] - lab_arr[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return np.sqrt(np.maximum(d2, 0.0))

def pick_reseed_candidate(sampled_lab, palette_lab, max_candidates=2000):
    N = sampled_lab.shape[0]
    if N == 0:
        return palette_lab[0].copy()
    if N > max_candidates:
        idx = np.random.choice(N, max_candidates, replace=False)
        cand = sampled_lab[idx]
    else:
        cand = sampled_lab
    diff = cand[:, None, :] - palette_lab[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    de = np.sqrt(np.maximum(d2, 0.0))
    min_de = np.min(de, axis=1)
    best = int(np.argmax(min_de))
    return cand[best].astype(np.float32)

def enforce_perceptual_separation(sampled_lab, palette_lab, counts, thresh=4.0, passes=2, max_candidates=2000):
    pal = palette_lab.copy()
    cnt = counts.copy() if counts is not None else np.ones((pal.shape[0],), dtype=np.int32)
    for _ in range(passes):
        de = delta_e76_matrix(pal)
        np.fill_diagonal(de, np.inf)
        i, j = np.unravel_index(np.argmin(de), de.shape)
        min_de = float(de[i, j])
        if min_de >= thresh:
            break
        if cnt[i] >= cnt[j]:
            keep, drop = i, j
        else:
            keep, drop = j, i
        new_color = pick_reseed_candidate(sampled_lab, pal, max_candidates=max_candidates)
        pal[drop] = new_color
        cnt[drop] = 0
    return pal

def kmeans_lab(pixels_rgb, K, max_iter=10, seed=1):
    # Stage 1: RGB cube binning
    img = Image.new("RGB", (len(pixels_rgb), 1))
    img.putdata(pixels_rgb)
    initc, initn = build_rgb_cubes(img, bins=16, count_threshold=1)
    initialized_palette = initial_palette_generation(initc, initn, K=K)
    # Stage 2: Resize and sample
    # (for single-row image, skip resize, just sample)
    sampled_pixels = pixels_rgb
    # Stage 2: Fast K-Means in CIELAB
    palette_rgb, mse_hist, palette_lab, counts = fast_kmeans_refine_lab(
        sampled_pixels, initialized_palette, max_iter=max_iter
    )
    # Stage 3: Perceptual separation
    if palette_lab is not None:
        sampled_lab = rgb_list_to_lab_array(sampled_pixels)
        fixed_palette_lab = enforce_perceptual_separation(
            sampled_lab=sampled_lab,
            palette_lab=palette_lab,
            counts=counts,
            thresh=DELTA_E_THRESH,
            passes=SOP3_PASSES,
            max_candidates=SOP3_CANDIDATES
        )
        final_palette_rgb = lab_array_to_rgb_list(fixed_palette_lab)
    else:
        final_palette_rgb = palette_rgb
    final_mse = mse_hist[-1] if mse_hist else 0.0
    return final_palette_rgb, fixed_palette_lab if palette_lab is not None else None, final_mse

def quantize_image_lab(img, centroids_lab, palette_rgb):
    arr_rgb = np.array(img, dtype=np.float32) / 255.0
    lab_img = rgb2lab(arr_rgb)
    h, w, _ = lab_img.shape
    flat_lab = lab_img.reshape(-1, 3)
    pal_lab = np.array(centroids_lab, dtype=np.float32)
    dists = np.sum((flat_lab[:, None, :] - pal_lab[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    palette_arr = np.array(palette_rgb, dtype=np.float32) / 255.0
    quantized_flat = palette_arr[labels]
    quantized_img = quantized_flat.reshape(h, w, 3)
    return Image.fromarray(np.clip(quantized_img * 255, 0, 255).astype(np.uint8))

def rgb_cube_kmeans_palette(img, num_colors, cube_bins=16, count_threshold=1, sample_rate=1.0, max_iter=10):
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
    img = img.convert("RGB")
    initc, initn = build_rgb_cubes(img, bins=cube_bins, count_threshold=count_threshold)
    initial_palette = initial_palette_generation(initc, initn, K=num_colors)
    sampled_pixels = block_sample_pixels(img, sample_rate)
    final_palette, mse = fast_kmeans_palette_refinement(sampled_pixels, initial_palette, max_iter=max_iter)
    return final_palette, mse


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