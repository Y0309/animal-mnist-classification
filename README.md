# Animal Image Classification: Logistic Regression, MLP, and CNN

A MATH 4036 Introduction to Machine Learning group project by **Jiyao Yin** and **Tewodros Tadesse**, comparing three models on the Animal-MNIST image-classification task.

## Project question

**How do different models and levels of complexity affect animal image-classification performance?**

We compare:

- **Logistic Regression (LR):** a traditional baseline using flattened image pixels.
- **Multilayer Perceptron (MLP):** a feed-forward neural network that learns nonlinear relationships between pixel features.
- **Convolutional Neural Network (CNN):** a neural network that preserves image structure and learns spatial features.

All three models use the same official training, validation, and test split.

## Dataset

Animal-MNIST contains 10,000 grayscale images at 28 × 28 pixels, across ten balanced classes (1,000 images per class):

`Bear · Bird · Cat · Cow · Dog · Elephant · Giraffe · Horse · Sheep · Zebra`

The official split used in this project is:

| Split | Images |
|---|---:|
| Training | 7,200 |
| Validation | 800 |
| Test | 2,000 |

The dataset is **not included** in this repository. The script loads it through the `animal-mnist` Python package. The project uses `animal-mnist==0.5.1`, which provides `load_animal_mnist_splits()` for the official split indices.

Dataset source: [SinaSR-6/Animal-MNIST](https://github.com/SinaSR-6/Animal-MNIST)

## Methods

### Preprocessing

- Pixel values are normalized to the range `[0, 1]` by dividing by 255.
- LR and MLP receive each 28 × 28 image as a flattened vector of 784 values.
- CNN receives the image as a `1 × 28 × 28` tensor, preserving the grayscale channel and spatial layout.

### Model configurations

| Model | Architecture / configuration | Training setup |
|---|---|---|
| Logistic Regression | 784 input features; no hidden layers | Scikit-learn, `max_iter=1000` |
| MLP | 784 → 256 → 128 → 10; ReLU in both hidden layers | Cross-entropy, Adam, learning rate 0.001, 200 full-batch epochs |
| CNN | Two 3 × 3 convolution layers (32 then 64 filters), ReLU and 2 × 2 max pooling; flatten 64 × 7 × 7 to 3,136; fully connected 128; dropout 0.3; 10 outputs | Cross-entropy, Adam, learning rate 0.001, batch size 64, maximum 30 epochs, early-stopping patience 6 |

The run uses random seed 42. The CNN restores the checkpoint with the highest validation accuracy.

## Reported test results

The following values are transcribed from the project's report (macro-averaged precision, recall, and F1):

| Model | Accuracy | Precision | Recall | Macro-F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 77.30% | 77.44% | 77.30% | 77.32% |
| MLP | 88.60% | 88.63% | 88.60% | 88.59% |
| CNN | 96.30% | 96.32% | 96.30% | 96.29% |

These are **paper-reported** values, not a claim that the script has been rerun in every environment. Run the experiment below to regenerate the metrics and figures from the installed package and current environment. Check the generated `results/model_comparison_results.csv` before quoting results.

> **Result-check note:** The report's prose also says that the CNN correctly classified 1,925 of 2,000 test images. That count corresponds to 96.25%, while the report's results table states 96.30%. This repository preserves the table values above as reported, but the final number should be confirmed from a fresh run before external use.

## Repository contents

```text
animal-mnist-classification/
├── main.py
├── requirements.txt
├── README.md
├── AI_USE.md
├── .gitignore
├── paper/
│   └── animal_mnist_classification_report.pdf
└── results/
    ├── paper_reported_test_metrics.csv
    └── paper_reported_test_accuracy.png
```

When `main.py` runs, it creates additional files in `results/`, including:

- `model_comparison_results.csv`
- `test_accuracy_comparison.png`
- `per_class_accuracy.csv` and `per_class_accuracy_comparison.png`
- Confusion matrices and classification reports for LR, MLP, and CNN
- MLP/CNN training-loss plots and the CNN validation-accuracy plot
- `test_predictions.csv`
- Example images from all ten classes

## Setup and run (Windows PowerShell)

Open PowerShell in the repository folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks virtual-environment activation, you can run this once in the current PowerShell window and then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The script prints the evaluation metrics and saves the tables, prediction files, and figures under `results/`. Plot windows may open during execution.

## Reproducibility

- Main run command: `python main.py`
- Random seed: `42`
- Dataset package: `animal-mnist==0.5.1`
- Official split: 7,200 / 800 / 2,000 train/validation/test images
- The test set is used for final reporting; the CNN validation set is used to select the best checkpoint.

For a fully pinned environment, record the exact versions installed on the machine used for the final run with:

```powershell
pip freeze > environment-lock.txt
```

## Interpretation and limitations

On this dataset, the reported comparison is LR < MLP < CNN. The results are consistent with the benefit of learning nonlinear and spatial image features, but the experiment does **not** isolate model complexity as the only cause: the three models also differ in architecture and input representation.

Limitations include one small grayscale dataset, only three model implementations, no data augmentation, and no cross-validation or systematic CNN hyperparameter search. These results should not be interpreted as proof that CNNs always outperform other models.

## Citation

Please cite the Animal-MNIST dataset and acknowledge its authors when reusing it. See the [dataset repository](https://github.com/SinaSR-6/Animal-MNIST) for its current citation guidance.

## Course context

SC/MATH 4036 3.00 — Introduction to Machine Learning  
Project route: **Cool ML Build**
