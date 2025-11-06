#!/usr/bin/env python3

"""
What this script does:
  1) Load the image and convert to grayscale in [0,1]
  2) Mild blur + Otsu threshold to isolate dark ink pixels
  3) Remove small specks
  4) Split the image into top and bottom halves (two plates)
  5) In each half: label connected components, skeletonize each, build an
     8-neighbor weighted graph on the skeleton pixels, and find the longest
     geodesic path between endpoints (the "main root")
  6) Filter components to keep only "root-like" ones (long, thin, elongated)
  7) Save a CSV of lengths (in original pixels) + an overlay PNG of the main paths, labeled in order of left to right in the image


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, skeletonize
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.transform import resize
import argparse

from pathlib import Path


# ---------------- Parameters ----------------



MIN_SIZE_GLOBAL = 80
GAUSS_SIGMA = 1.0

# ---------------- Utilities ----------------
def to_gray01(a):
    if a.ndim == 2:
        g = a.astype(np.float32)
        if g.max() > 1:
            g /= g.max()
        return g
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    return rgb2gray(a)

def skel_to_graph(skel):
    ys, xs = np.where(skel)
    if len(ys) == 0:
        return None, (ys, xs)
    coords = list(zip(ys.tolist(), xs.tolist()))
    idx = {c: i for i, c in enumerate(coords)}
    rows, cols, data = [], [], []
    nbrs = [(-1,-1,np.sqrt(2)),(-1,0,1.0),(-1,1,np.sqrt(2)),
            (0,-1,1.0),(0,1,1.0),(1,-1,np.sqrt(2)),(1,0,1.0),(1,1,np.sqrt(2))]
    for i, (r,c) in enumerate(coords):
        for dr,dc,w in nbrs:
            j = idx.get((r+dr, c+dc))
            if j is not None:
                rows.append(i); cols.append(j); data.append(w)
    from scipy.sparse import coo_matrix
    G = coo_matrix((data, (rows, cols)), shape=(len(coords),)*2).tocsr()
    return G, (ys, xs)

def endpoints(G):
    import numpy as np
    deg = np.array(G.astype(bool).sum(axis=1)).ravel()
    return np.where(deg == 1)[0]

def longest_path(G):
    from scipy.sparse.csgraph import dijkstra
    e = endpoints(G)
    n = G.shape[0]
    if n == 0:
        return 0.0, None, None
    nodes = e if len(e) >= 2 else np.arange(n)
    best = (0.0, None, None)
    for s in nodes:
        d = dijkstra(G, directed=False, indices=s)
        mask = e if len(e) >= 2 else np.arange(n)
        valid = d[mask]
        if not np.isfinite(valid).any():
            continue
        t = mask[np.argmax(valid)]
        L = float(valid.max())
        if L > best[0]:
            best = (L, int(s), int(t))
    return best

def path_nodes(G, src, dst):
    from scipy.sparse.csgraph import dijkstra
    if src is None or dst is None:
        return []
    d, pred = dijkstra(G, directed=False, indices=src, return_predecessors=True)
    if not np.isfinite(d[dst]):
        return []
    path = [dst]
    cur = dst
    while cur != src and cur != -9999:
        cur = pred[cur]
        if cur == -9999:
            break
        path.append(cur)
    return path[::-1]

def filter_root_like(region, main_len_scaled, scale, img_dim):
    """
    Heuristics to keep 'root-like' components:
      - minimum geodesic length
      - thin average width = area / main_length
      - tall bounding box (high aspect ratio)
    All thresholds are expressed for the *scaled* image; we multiply by `scale`.
    """
    min_len = 80 * scale       # at scale=1: 80 px minimum main length
    max_width = 14 * scale     # average width upper bound
    min_ar = 3.0               # bbox aspect ratio (height/width) must be tall
    max_len = img_dim[0]/4

    h = region.bbox[2] - region.bbox[0]    # bbox height (scaled)
    w = region.bbox[3] - region.bbox[1]    # bbox width (scaled)
    if w <= 0: return False                # avoid div by zero
    ar = h / w                             # aspect ratio

    if main_len_scaled < min_len:          # must be long enough
        return False
    if main_len_scaled > max_len:          # can't be too long
        return False
    avg_width = region.area / max(main_len_scaled, 1e-6)  # area/length ~ mean width
    if avg_width > max_width:              # too thick -> probably debris/frame/text
        return False
    if ar < min_ar:                        # too stubby
        return False
    return True                            # passes all checks

# ---------------- Main ----------------

def process_image(IMG_PATH):
    file_name = IMG_PATH.stem
    orig = imread(input_dir / IMG_PATH)
    gray = to_gray01(orig)
    H, W = gray.shape
    target = 1600.0
    SCALE = min(1.0, target / max(H, W))
    if SCALE != 1.0:
        gray_s = resize(gray, (int(H*SCALE), int(W*SCALE)), order=1, anti_aliasing=True, preserve_range=True)
    else:
        gray_s = gray

    blur = gaussian(gray_s, sigma=GAUSS_SIGMA, preserve_range=True)
    thr = threshold_otsu(blur)
    fg = blur < thr
    fg = remove_small_objects(fg, min_size=int(MIN_SIZE_GLOBAL * SCALE))

    h2 = fg.shape[0] // 2
    halves = [("top", fg[:h2, :], (0, h2)), ("bottom", fg[h2:, :], (h2, fg.shape[0]))]

    records = []
    overlay_paths = []

    for half_name, mask_half, (row0, row1) in halves:
        mh = clear_border(mask_half)
        lab = label(mh, connectivity=2)
        regions = regionprops(lab)
        candidates = []

        for r in regions:
            comp = (lab == r.label)
            skel = skeletonize(comp)
            G, (ys, xs) = skel_to_graph(skel)
            if G is None or G.shape[0] == 0:
                continue
            L_scaled, s, t = longest_path(G)

            if not filter_root_like(r, L_scaled, 1.0, (H,W)):  # apply shape/length filters
                continue
            pnodes = path_nodes(G, s, t)
            if not pnodes:
                continue
            path_pix = np.array([(ys[i], xs[i]) for i in pnodes], dtype=int)
            path_pix[:,0] += row0
            L_full = L_scaled / SCALE
            x_median = float(np.median(path_pix[:,1]))
            candidates.append((r, L_full, path_pix, x_median))





        # sort left->right by median x
        candidates.sort(key=lambda x: x[3])

        for idx, (r, L_full, path_pix, x_med) in enumerate(candidates, start=1):
            records.append({
                "half": half_name,
                "rank_in_half": idx,
                "main_length_px": float(L_full),
                "bbox_min_row": int(r.bbox[0] / SCALE + (row0 / SCALE)),
                "bbox_min_col": int(r.bbox[1] / SCALE),
                "bbox_max_row": int(r.bbox[2] / SCALE + (row0 / SCALE)),
                "bbox_max_col": int(r.bbox[3] / SCALE),
            })
            overlay_paths.append((f"{half_name}-{idx}", path_pix))

    # write csv
    #CSV_PATH = os.path.join(args.output_dir, f"{file_name}_measurements.csv")
    CSV_PATH = output_dir / f"{file_name}_measurements.csv"
    df = pd.DataFrame.from_records(records).sort_values(["half", "rank_in_half"]).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved CSV: {CSV_PATH}")

    # overlay
    OVERLAY_PATH = output_dir / f"{file_name}_overlay.png"
    #OVERLAY_PATH = os.path.join(args.output_dir, f"{file_name}_overlay.png")
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(gray_s, cmap="gray")
    for name, path in overlay_paths:
        if len(path) == 0:
            continue
        ax.plot(path[:,1], path[:,0], linewidth=2)
        ax.text(path[0,1], path[0,0], name, fontsize=6)
    ax.set_axis_off()
    ax.set_title("Main paths (labeled left→right within each half)")
    fig.savefig(OVERLAY_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay: {OVERLAY_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory with input segmented images")
    parser.add_argument("--output_dir_name", default="measure_roots_output", help="Directory name to save outputs — will be created in the parent folder of input_dir.")
    parser.add_argument("--dpi", type=int, default=600, help="DPI of input images. Default is 600.")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir_name = args.output_dir_name
    output_dir = Path.resolve(Path(args.input_dir).parent) / Path(output_dir_name)
    output_dir.mkdir(exist_ok=True)

    for img in input_dir.iterdir():
        if img.suffix not in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
            continue
        process_image(img)