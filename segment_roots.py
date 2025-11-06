#!/usr/bin/env python3

# basics
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# image processing
from skimage import feature, color, filters, io, util
from skimage.morphology import dilation, footprint_rectangle, closing, opening, disk
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize
from skimage.segmentation import clear_border
from scipy import ndimage
import argparse



# ---------------- Display utils -------


def show(img, ax=None, cmap="gray", figsize=None):
    if ax is None:
        fig = plt.gcf()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
    ax.set_axis_off()
    ax.imshow(img, cmap=cmap)
    if figsize is not None:
        fig.set_size_inches(figsize)

def showmosaic(img1, img2, cmap1="gray", cmap2="BuGn"):
    fig = plt.figure()
    fig.set_size_inches(16.5, 10.5)

    ax1 = plt.Axes(fig, [0., 0., 0.475, 1.])
    fig.add_axes(ax1)
    show(img1, ax1, cmap=cmap1)

    ax2 = plt.Axes(fig, [0.5, 0., 0.475, 1.])
    fig.add_axes(ax2)
    show(img2, ax2, cmap=cmap2)

# ---------------- Processing utils -------

def load_img(img_path, xslice=(0,-1), yslice=(0,-1)):
    # load full image
    img = io.imread(img_path)
    try: # show image only works with uint8, not uint16
      img_converted = util.img_as_ubyte(img)
    except ValueError: # image is already uint8
      img_converted = img

    # return only cropped area (if any)
    return img_converted[xslice[0]:xslice[1],
               yslice[0]:yslice[1]]

def load_gray_img(img_path, xslice=(0,-1), yslice=(0,-1)):
    img = load_img(img_path, xslice, yslice)
    try:
        gray_img = color.rgb2gray(img)
    except ValueError:  # image is already gray
        gray_img = img

    return gray_img

def get_green_mask(img):
    # break down image into components
    ##hsv = color.rgb2hsv(img)
    h = img[..., 0].astype(float)
    s = img[..., 1].astype(float)
    v = img[..., 2].astype(float)

    # pull out interactive thresholds
    hrange = hue_range
    srange = [18, 255]
    vrange = [0, 171]

    # use interactively determined thresholds
    mask = (h >= hrange[0]) & (h <= hrange[1]) & \
           (s >= srange[0]) & (s <= srange[1]) & \
           (v >= vrange[0]) & (v <= vrange[1])

    return mask

def apply_canny(img, sigma, high, low):
    # smooth image
    smoothed = filters.gaussian(img, sigma=sigma, preserve_range=True)
    # calculate gradient magnitudes
    gx = ndimage.gaussian_filter(smoothed, sigma=sigma, order=[1,0])
    gy = ndimage.gaussian_filter(smoothed, sigma=sigma, order=[0,1])
    grad_mag = np.hypot(gx, gy)
    # calculate thresholds as percentiles on gradient magnitude
    hi = np.percentile(grad_mag, high)
    lo = np.percentile(grad_mag, low)
    # get edges based on Canny
    edges = feature.canny(img, sigma=sigma,
                          low_threshold=lo, high_threshold=hi)
    return edges


def filter_by_length(edges):
    # dilate edges
    connected_edges = dilation(edges, footprint_rectangle((3,3)))
    #connected_edges = dilation(edges, footprint_rectangle((1,1)))
    # label connected components
    label_img = label(connected_edges, connectivity=2)
    # get mask
    mask = remove_small_objects(label_img, min_size=1000) > 0

    return mask

def close(mask):
    return closing(mask, footprint_rectangle((7,7)))

def invert(mask):
    return ~mask

def process(img, sigma, high, low):
    edges = apply_canny(img, sigma, high, low)
    filtered = filter_by_length(edges)
    closed = close(filtered)
    inverted = invert(closed)

    return inverted


def process_batch(image_directory, targetdir, sigma, high, low):

    targetdir = Path(targetdir)
    targetdir.mkdir(exist_ok=True)

    segmented_seedling_dir = targetdir / Path("segmented_seedling")
    segmented_seedling_dir.mkdir(exist_ok=True)
    segmented_shoot_dir = targetdir / Path("segmented_shoot")   
    segmented_shoot_dir.mkdir(exist_ok=True)
    segmented_root_dir = targetdir / Path("segmented_root")
    segmented_root_dir.mkdir(exist_ok=True)


    i = 1
    for file in Path(image_directory).iterdir():
        if file.suffix not in ('.tif', '.tiff'):
            continue

        print(f"[{i}] Processing {file.name}")

        # Segment whole seedling
        gray_img = load_gray_img(file)
        seedling_segment   = process(gray_img, sigma, high, low)
        seedling_file_path = segmented_seedling_dir / f"{file.stem}.tif"
        plt.imsave(seedling_file_path, seedling_segment, format='tiff', dpi=600, cmap="gray")

        # Segment shoot
        color_img  = load_img(file)
        green_mask = get_green_mask(color_img)
        filtered_green_mask = filter_by_length(green_mask)
        dilated_shoot = dilation(filtered_green_mask, footprint_rectangle((10, 10)))
        shoot_file_path = segmented_shoot_dir / f"{file.stem}.tif"
        plt.imsave(shoot_file_path, dilated_shoot, format='tiff', dpi=600, cmap="BuGn")

        # Subtract shoot to get just root
        subtract_shoot = np.logical_or(seedling_segment, dilated_shoot)
        root_file_path = segmented_root_dir / f"{file.stem}.tif"
        plt.imsave(root_file_path, subtract_shoot, format='tiff', dpi=600, cmap="gray")

        i += 1

    print("Done!")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="The only required argument is the input directory \
                                     containing images to be segmented. The full input directory path should be given (e.g. /Users/.../images).")

    parser.add_argument("img_dir", help="directory with input images to be segmented")
    parser.add_argument("--output_dir_name", default="segmented_output", help="Directory name to save outputs — will be created in the parent folder of input_dir")
    parser.add_argument("--dpi", type=int, default=600, help="DPI of output segmented images. Default is 600.")
    parser.add_argument("--hue_range", type=str, default="39,115", help="[low,high] threshold range for hue, used in shoot masking. Default is (39,115).")
        # used to be [39, 108]
    parser.add_argument("--sigma", type=float, default=2.0, help="Sigma for edge detection with canny. Default is 2.0.")
    parser.add_argument("--low_thresh", type=int, default=98, help="Lower bound for edge detection with canny. Default is 98.")
    parser.add_argument("--high_thresh", type=int, default=100, help="Upper bound for edge detection with canny. Default is 100.")
    args = parser.parse_args()

    hue_range = [int(x) for x in args.hue_range.split(",")]
    sigma = args.sigma
    low_thresh = args.low_thresh
    high_thresh = args.high_thresh
        
    
    output_dir_name = args.output_dir_name
    output_dir = Path.resolve(Path(args.input_dir).parent) / Path(output_dir_name)
    process_batch(args.img_dir, output_dir, sigma, high_thresh, low_thresh)