# CNN Image Classification with Keras (Binary Classifier)

A beginner-friendly Jupyter notebook that walks through the full pipeline of preparing an image folder and training a small **Convolutional Neural Network (CNN)** with TensorFlow / Keras.

> ⚠️ **Note:** The labels in this notebook are **randomly generated placeholders** (`np.random.choice([0, 1])`), so the reported accuracy is **not** a meaningful result. The notebook is a template for the workflow. To get a real classifier, replace the random labels with real ones (see [Using your own labels](#using-your-own-labels)).

## What the notebook does

1. **Loads images** from a folder (`/content/Images`, 121 images in the original run).
2. **Explores the data**: inspects array shapes and pixel values, and displays random samples in color and grayscale.
3. **Checks image sizes** (the dataset contains `179×179` and `224×224` images).
4. **Preprocesses** every image: grayscale conversion → resize to `100×120` → reshape to add a channel dimension.
5. **Splits the data** into train (75%) and test (25%) sets using `train_test_split`.
6. **Builds a CNN** with the architecture below.
7. **Compiles and trains** the model for 10 epochs.

## Model architecture

| # | Layer | Details |
|---|-------|---------|
| 1 | `Conv2D` | 32 filters, 3×3, valid padding |
| 2 | `MaxPooling2D` | 2×2 |
| 3 | `BatchNormalization` | |
| 4 | `Conv2D` | 32 filters, 4×4, valid padding |
| 5 | `MaxPooling2D` | 2×2 |
| 6 | `BatchNormalization` | |
| 7 | `Flatten` | |
| 8 | `Dense` | 128 units, ReLU |
| 9 | `Dropout` | 0.2 |
| 10 | `Dense` | 1 unit, Sigmoid (binary output) |

- **Optimizer:** Adam
- **Loss:** Binary cross-entropy
- **Metric:** Accuracy
- **Input shape:** `(120, 100, 1)`

## Requirements

- Python 3.9+
- TensorFlow / Keras
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- scikit-learn

Install everything with:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

## How to run

### Option 1: Google Colab (recommended)
1. Upload `CNN.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Upload your images into a folder named `Images` in the Colab file browser (so the path is `/content/Images`).
3. Run all cells.

### Option 2: Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Put your images in a folder, then change this line in the notebook to point to it:
   ```python
   ImagesPath = 'path/to/your/Images'
   ```
3. Launch Jupyter and run the notebook:
   ```bash
   jupyter notebook CNN.ipynb
   ```

## Using your own labels

Replace this cell:

```python
y = [np.random.choice([0, 1]) for i in range(121)]
```

with real labels that match the order of `os.listdir(ImagesPath)`. For example, build them from file names or from a CSV file. Make sure the number of labels equals the number of images.

## Known issues / possible improvements

- **Random labels:** see the note at the top.
- **Reshape order:** after `cv2.resize(img, (H, W))` the image has shape `(100, 120)`, but it is then reshaped to `(H, W, 1)` = `(120, 100, 1)`. This scrambles the pixel layout. Use `img.reshape(W, H, 1)` and `input_shape=(W, H, 1)` instead, or use `np.expand_dims(img, -1)`.
- **No normalization:** divide pixel values by 255 (`X / 255.0`) before training.
- **No validation during training:** pass `validation_data=(X_test, y_test)` to `fit()` and evaluate with `model.evaluate(...)`.
- **Small dataset:** consider data augmentation (`ImageDataGenerator` or Keras preprocessing layers) and more data.
- **Hard-coded path and image count:** use `len(Images)` instead of `121`.

## Project structure

```
.
├── CNN.ipynb      # Main notebook
├── README.md      # This file
└── Images/        # (Not included) your image dataset
```

## License

Add a license of your choice (e.g., MIT) as a `LICENSE` file.

