# RootSeg

## Setup
First, install the required dependencies. Make sure you have at least Python 3.7. It is recommended to create a virtual environment first.

```bash
pip install -r requirements.txt
```

Create an image directory containing all the images to be processed. They should have .tif or .tiff suffixes. 

## Usage

To use RootSeg, `cd` into the folder containing this repo. 

### Step 1: Segmentation
First, segment the image:

```bash
python3 segment_roots.py img_dir
```
The full input directory path for `img_dir` should be given (e.g. `/Users/.../images`).

Optional arguments:


  `--output_dir_name OUTPUT_DIR_NAME`
                        Directory name to save outputs — will be created in the parent folder of input_dir
  `--dpi DPI`             DPI of output segmented images. Default is 600.
  `--hue_range HUE_RANGE`
                        [low,high] threshold range for hue, used in shoot masking. Default is (39,115).
  `--sigma SIGMA`         Sigma for edge detection with canny. Default is 2.0.
  `--low_thresh LOW_THRESH`
                        Lower bound for edge detection with canny. Default is 98.
  `--high_thresh HIGH_THRESH`
                        Upper bound for edge detection with canny. Default is 100.

For each image, the output directory will contain three segmented images: `segmented_seedling`, `segmented_shoot`, `segmented_root`

### Step 2: Measuring roots

If you wish to measure the length of the roots, run:

```bash
python3 measure_roots.py input_dir
```
You can directly use the `segmented_root` directory generated in Step 1 as the `input_dir`. 

Optional arguments:

  `--output_dir_name OUTPUT_DIR_NAME`
                        Directory name to save outputs — will be created in the parent folder of input_dir.
  `--dpi DPI`             DPI of input images. Default is 600.

For each image, the output directory will contain a CSV of all measured roots and an overlay image with the corresponding labels and traces for each root.
