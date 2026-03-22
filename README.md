# Deep Learning Multi-Classification Using Chest X-Ray Images

A deep learning project for multi-class classification of chest X-ray images to diagnose COVID-19, Normal, Pneumonia, and Tuberculosis conditions using DenseNet121 architecture with k-fold cross-validation.

## Overview

This project implements a robust multi-class classification system that analyzes chest X-ray images to classify patients into four categories:
- **COVID-19**: Coronavirus disease cases
- **Normal**: Healthy patients with no detected conditions
- **Pneumonia**: Patients with pneumonia
- **Tuberculosis**: Patients with tuberculosis

The model is trained using DenseNet121, a pre-trained convolutional neural network, enhanced with custom layers and evaluated using k-fold cross-validation for robust performance assessment.

## Features

- **DenseNet121 Architecture**: Leverages pre-trained ImageNet weights for transfer learning
- **K-Fold Cross-Validation**: 5-fold cross-validation for robust model evaluation
- **Data Augmentation**: ImageDataGenerator for improved generalization
- **Class Imbalance Handling**: Stratified sampling for balanced training
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, AUC-ROC
- **Early Stopping & Learning Rate Reduction**: Prevents overfitting and optimizes convergence

## Project Structure

```
.
├── README.md                    # This file
├── main.ipynb                  # Main prediction script
├── trained_models/             # Pre-trained model weights
│   ├── Model_fold_1.h5
│   ├── Model_fold_2.h5
│   ├── Model_fold_3.h5
│   ├── Model_fold_4.h5
│   └── Model_fold_5.h5
├── requirements.txt            # Python dependencies
├── Documentation.pdf           # Detailed project documentation
└── LICENSE                      # MIT License
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kylerimpos/Deep-Learning-Multi-Classification-Using-Chest-X-Ray-Images.git
cd Deep-Learning-Multi-Classification-Using-Chest-X-Ray-Images
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Predictions

Open and run `main.ipynb` in Jupyter Notebook:
```bash
jupyter notebook main.ipynb
```

The notebook includes:
- Dataset loading and exploration
- Model inference using pre-trained weights
- Prediction visualization
- Performance metrics calculation

### Using Pre-trained Models

The `trained_models/` directory contains 5 pre-trained model weights (one for each fold). Load and use them as follows:

```python
from tensorflow.keras.models import load_model

# Load a specific fold's model
model = load_model('trained_models/Model_fold_1.h5')

# Make predictions
predictions = model.predict(x_test)
```

## Dataset

The project requires a `DATASET/` directory with the following structure:
```
DATASET/
├── Covid/
├── Normal/
├── Pneumonia/
└── Tuberculosis/
```

Each folder should contain corresponding chest X-ray images in PNG or JPG format.

## Model Architecture

- **Base Model**: DenseNet121 (pre-trained on ImageNet)
- **Custom Layers**:
  - Global Average Pooling
  - Batch Normalization
  - Dense (512 units, ReLU activation)
  - Dropout (0.5)
  - Output Dense Layer (4 units, softmax activation)

## Training Details

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall, AUC-ROC, F1-Score
- **Validation Strategy**: 5-Fold Stratified Cross-Validation
- **Batch Size**: Configurable (default: 32)
- **Early Stopping**: Patience = 10 epochs
- **Learning Rate**: Initial reduction on validation loss plateau

## Results

The model achieves robust performance across all four classes using k-fold cross-validation for reliable evaluation. See `Documentation.pdf` for detailed results and analysis.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@project{chestxray_classification,
  title={Deep Learning Multi-Classification Using Chest X-Ray Images},
  author={Rimpos, Kyle},
  year={2024}
}
```

## Contact

For questions or suggestions, please open an issue on GitHub.

## Acknowledgments

- DenseNet121 architecture: Huang et al., 2017
- TensorFlow and Keras teams for excellent deep learning frameworks
- The chest X-ray dataset community
