# Floor Detection using SVM — Three Methods

**Made with love by C Murali Madhav**

This project implements **floor vs. non-floor** pixel/region classification on indoor images using the **CMM dataset** and **Support Vector Machines (SVM)**. Three feature-engineering approaches are compared: RGB-only pixels, RGB plus spatial coordinates, and KMeans-based region-level features.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methods (Detailed)](#methods-detailed)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [License](#license)

---

## Overview

The goal is to **classify image regions as floor or non-floor** using ground-truth annotations in COCO format. The notebook:

1. Loads images and builds binary floor masks from polygon segmentations.
2. Applies **three feature-extraction methods** (pixel-level and region-level).
3. Trains an **RBF-kernel SVM** for each method with an 80/20 stratified train/test split.
4. Compares **accuracy**, **training time**, and **visual predictions** across methods.

**Summary of results:** Method 3 (KMeans region-level) achieves the highest test accuracy (~92.5%) and the fastest training; Method 2 (RGB + spatial) is second (~86%); Method 1 (RGB only) is baseline (~78%).

---

## Dataset

- **Source:** CMM dataset (custom indoor images with floor annotations).
- **Annotations:** COCO-style JSON at `dataset/CMM_Annotations.json` (polygon segmentations per image).
- **Images:** Stored in `dataset/` (e.g. `IMG_9490.jpeg`, …). Only images that have at least one floor annotation are used.
- **Processing:**
  - Each image is resized so the longer side is at most **512 pixels** (configurable) to keep memory and speed manageable.
  - Binary masks are built by drawing polygon segmentations onto a blank canvas (floor = 1, non-floor = 0).
- **Usage:** 12 image–mask pairs are loaded. All three methods use the **bottom half** of each image (typical floor region in indoor shots) for pixel-based methods; Method 3 uses the full image for KMeans clustering.

---

## Methods (Detailed)

### Method 1: RGB only (bottom-half pixels)

**Idea:** Use only **color** to decide if a pixel is floor or not. No information about where the pixel is in the image.

**Feature extraction:**

- For each image, take only the **bottom half** (rows from `H/2` to `H`).
- For each pixel in that half: features = **[R, G, B]** (3 values). Label = value from the ground-truth mask (0 = non-floor, 1 = floor).
- To limit dataset size, pixels are **subsampled** to at most **5,000 per image** (random, without replacement).

**Classifier:** RBF SVM (`C=10`, `gamma='scale'`, `class_weight='balanced'`). Features are standardized with `StandardScaler` before training.

**Interpretation:** Purely color-based; the model learns which RGB values tend to be floor (e.g. grey/brown) vs non-floor. It cannot use the fact that floor is usually in the lower part of the image, so accuracy is limited.

---

### Method 2: RGB + normalized (x, y) (bottom-half pixels)

**Idea:** Add **position in the image** so the model can use both color and location (e.g. “floor is often at the bottom”).

**Feature extraction:**

- Same as Method 1 (bottom half, optional subsample to 5,000 pixels per image).
- For each pixel, features = **[R, G, B, x_norm, y_norm]** (5 values), where:
  - **x_norm** = column index / image width  (0 at left, 1 at right).
  - **y_norm** = row index / image height    (0 at top, 1 at bottom).

**Classifier:** Same SVM and scaling as Method 1.

**Interpretation:** The SVM can learn rules like “darker, low y_norm” → floor. This typically improves accuracy over Method 1 because floor regions are often in the lower part of the frame.

---

### Method 3: KMeans region-level features

**Idea:** Instead of classifying every pixel, **group pixels into regions** with KMeans, then describe each region by **mean color and centroid**, and classify **regions** as floor or non-floor. This reduces noise and the number of samples, and often improves accuracy and speed.

**Feature extraction:**

1. **KMeans clustering:** All pixels of the image (full frame) are represented by **(R, G, B, x_norm, y_norm)**. KMeans with **n_clusters = 100** is run to get a label per pixel (region id).
2. **Per-region features and labels:**
   - For each cluster/region, compute:
     - **mean R, G, B** over all pixels in that region;
     - **centroid** in normalized coordinates: (cx_norm, cy_norm).
   - Feature vector per region = **[mean_R, mean_G, mean_B, cx_norm, cy_norm]** (5 values).
   - Label for the region = **majority vote** of the ground-truth mask over the region’s pixels (threshold 0.5 → 0 or 1).
3. All regions from all images are collected into one dataset (e.g. 12 images × 100 regions = 1,200 samples for training + test).

**Classifier:** Same RBF SVM and scaling. Training is very fast because the number of samples is small (region count, not pixel count).

**Interpretation:** The model learns which **region-level** color and position combinations correspond to floor (e.g. large, low-lying grey/brown regions). Predictions are then applied per region and mapped back to a pixel mask for visualization.

---

## Results

Reported from the notebook (stratified 80/20 split, same random seed):

| Method              | Test accuracy | Training time (approx.) |
|---------------------|---------------|--------------------------|
| 1 — RGB only        | **~78.0%**    | ~30.5 s                  |
| 2 — RGB + (x, y)    | **~86.1%**    | ~18.5 s                  |
| 3 — KMeans regions  | **~92.5%**    | ~0.008 s                 |

- **Best accuracy:** Method 3 (KMeans region-level).
- **Fastest training:** Method 3 (far fewer samples).
- **Method 2** improves over Method 1 by adding spatial context; **Method 3** improves further by working at region level and using both color and position.

The notebook also includes:

- Confusion matrices and classification reports (precision, recall, F1) per method.
- Visualizations: ground-truth masks, predicted masks, and overlays for sample images.
- A side-by-side comparison of the three predicted masks on the same image.

---

## Project Structure

```
Floor_Detection_Svm/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── Floor_detection.ipynb     # Main notebook (all three methods + results)
├── dataset/
│   ├── CMM_Annotations.json  # COCO-format floor annotations
│   └── IMG_*.jpeg            # Images
```

---

## Requirements

- **Python:** 3.x (tested with 3.11).
- **Libraries:**  
  `numpy`, `opencv-python` (cv2), `matplotlib`, `scikit-image`, `scikit-learn`, `PIL` (Pillow).

Install with:

```bash
pip install numpy opencv-python matplotlib scikit-image scikit-learn Pillow
```

(Use the correct package name **scikit-image**, not `skimage`, for installation.)

---

## How to Run

1. Clone or download this repository and ensure `dataset/CMM_Annotations.json` and the images in `dataset/` are present.
2. Open `Floor_detection.ipynb` in Jupyter or a compatible environment.
3. Run all cells in order. The notebook will:
   - Load and resize images and build masks.
   - Run Methods 1, 2, and 3 (feature extraction → train/test split → scale → train SVM → evaluate).
   - Print metrics and plots (confusion matrices, prediction overlays, comparison figures).

No command-line arguments are required; paths are relative to the project root.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for the full text.

---

**Made with love by C Murali Madhav**
