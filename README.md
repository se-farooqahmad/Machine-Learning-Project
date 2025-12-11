# Face Recognition Final Project

A machine learning project for face recognition using Principal Component Analysis (PCA) and deep learning techniques on the Olivetti Faces dataset.

## Project Structure

- **Task-01/**: PCA-based face recognition
  - PCA.py: Principal Component Analysis implementation
  - olivetti_faces.npy: Dataset file
  - olivetti_faces_target.npy: Target labels

- **Task-02/**: Deep Learning approaches
  - DeepLearning.py: Deep learning model implementation
  - DeepLearning_to_train.py: Training script
  - 	est1.jpg: Test image

- **Dataset and Program/**: Complete project implementation
  - FinalProj.py: Final project code combining all approaches
  - olivetti_faces.npy: Dataset file
  - olivetti_faces_target.npy: Target labels

## Dataset

The project uses the **Olivetti Faces Dataset**, which contains:
- 400 face images
- 40 distinct people (10 images per person)
- 64x64 pixel grayscale images

## Requirements

- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib
- opencv-python (for some tasks)

## Installation

\\\Bash
pip install -r requirements.txt
\\\

## Usage

### Task-01: PCA-based Recognition
\\\Bash
python Task-01/PCA.py
\\\

### Task-02: Deep Learning
\\\Bash
python Task-02/DeepLearning.py
\\\

### Final Project
\\\Bash
python "Dataset and Program/FinalProj.py"
\\\

## Techniques Used

1. **Principal Component Analysis (PCA)**: Dimensionality reduction for face feature extraction
2. **Logistic Regression**: Classification on PCA features
3. **Support Vector Machine (SVM)**: Classification with kernel methods
4. **Deep Learning**: Neural networks for feature learning and classification

## Results

The models are evaluated using:
- Accuracy metrics
- Confusion matrices
- Classification reports

## Notes

- Ensure the .npy data files are in the same directory as the Python scripts
- Images are normalized to [0, 1] interval before processing
- The dataset is split into 70% training and 30% testing sets

## Author

Farooq Ahmad 17L-4407

## License

This project is for educational purposes.
