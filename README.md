# Hand Gesture Classification using MediaPipe Landmarks

A machine learning project for classifying static hand gestures from MediaPipe hand landmarks (`x, y, z`) and running real-time inference from a webcam feed.

## Project Overview

This repository contains:

- A full experimentation notebook for data exploration, preprocessing, model training, and evaluation.
- A labeled landmark dataset with 18 gesture classes.
- Saved trained models (SVC, Random Forest, XGBoost).
- A real-time inference script using OpenCV + MediaPipe.
- Confusion matrix visualizations for multiple model configurations.

The best-performing model in the notebook experiments is an RBF SVC with `C=10`, which is also the default model loaded by the inference script.

## Repository Structure

```text
.
├── hand_gesture_classificaiton.ipynb        # End-to-end training and analysis notebook
├── hand_landmarks_data.csv                  # Dataset (MediaPipe landmarks + label)
├── gesture_inference.py                     # Real-time webcam inference script
├── Models/
│   ├── gesture_svc.joblib                   # Best SVC model used in inference script
│   ├── gesture_random_forest.joblib         # Random Forest baseline model
│   └── gesture_xgboost.joblib               # XGBoost model bundle (model + label encoder)
└── Figures/                                 # Saved confusion matrices from experiments
```

## Dataset

- **File:** `hand_landmarks_data.csv`
- **Rows:** 25,675 samples
- **Columns:** 64 total
  - 63 numeric landmark features (`x1..z21`)
  - 1 target column: `label`
- **Number of classes:** 18

Gesture classes:

`call, dislike, fist, four, like, mute, ok, one, palm, peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted`

## Preprocessing

The same preprocessing logic is used in training and inference:

1. Re-center all `x, y` coordinates relative to the wrist point.
2. Scale `x, y` coordinates by the distance to middle finger tip landmark (`12`).
3. Keep `z` values unchanged.

This normalization improves invariance to translation and hand size in the image frame.

## Model Training Summary

The notebook evaluates several models and settings:

- Random Forest (`n_estimators = 20, 50, 100`)
- SVC (`linear`, `rbf`, `rbf + high C`)
- KNN (`k=5`)
- XGBoost (shallow and deep configurations)

Data split in notebook:

- Train: 80%
- Validation: 10%
- Test: 10%

### Best Test Macro F1-Score (from notebook outputs)

| Model | Test Macro F1-Score |
|---|---:|
| SVC (RBF, `gamma=1.0`, `C=10.0`) | **0.9880** |
| SVC (RBF, `gamma=1.0`, `C=1.0`) | 0.9841 |
| Random Forest (`n_estimators=100`) | 0.9789 |
| XGBoost (deep) | 0.9762 |
| KNN (`k=5`) | 0.9740 |
| Random Forest (`n_estimators=50`) | 0.9767 |
| Random Forest (`n_estimators=20`) | 0.9706 |
| XGBoost (shallow) | 0.9694 |
| SVC (linear) | 0.8521 |

## Real-Time Inference

`gesture_inference.py`:

- Captures webcam frames with OpenCV.
- Detects one hand using MediaPipe Hands.
- Extracts 21 landmarks (`x, y, z`) and preprocesses them.
- Runs classification with `Models/gesture_svc.joblib`.
- Overlays the prediction on the video stream.

### Run Inference

```bash
python gesture_inference.py
```

Press `q` to exit.

## Demo Video

> [Demo](https://github.com/user-attachments/assets/cee3a19b-db62-4ad8-a45b-781e6920cfd4)

- **Description:** Short walkthrough showing hand gesture detection and real-time predictions.

## Installation

Use Python 3.12+ (recommended) and install dependencies:

```bash
pip install numpy pandas opencv-python mediapipe scikit-learn xgboost joblib matplotlib
```

> Note: `xgboost` is required for notebook training/evaluation and loading the XGBoost model bundle.

## Reproducing Training

1. Open and run `hand_gesture_classificaiton.ipynb`.
2. Execute cells in order to:
   - Explore and preprocess data.
   - Train and compare models.
   - Generate confusion matrices in `Figures/`.
   - Save models to `Models/`.

## Notes

- The inference script currently displays raw predicted class output directly from the loaded model.
- For XGBoost inference, the saved bundle includes both model and label encoder (`{"model": ..., "encoder": ...}`).

## Contact
For questions or collaboration, feel free to connect:
[LinkedIn](https://www.linkedin.com/in/mostafa-mamdouh-80b110228)