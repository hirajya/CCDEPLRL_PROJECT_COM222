# Bruise Detection Using Convolutional Neural Networks

A deep learning-based medical image analysis system for automated bruise detection using Convolutional Neural Networks (CNNs). This project implements a binary classification model to distinguish between bruised and normal skin in medical images.

## 🔬 Project Overview

This research project develops and evaluates CNN architectures for automated bruise detection in medical images. The system uses advanced image preprocessing, data augmentation, and regularization techniques to achieve robust performance in clinical settings.

### Key Features

- **Binary Classification**: Distinguishes between bruised and normal skin
- **Advanced CNN Architecture**: Custom-designed network with optimized layers
- **Data Augmentation**: Robust training with image transformations
- **Regularization Techniques**: L2 regularization and dropout for improved generalization
- **Web Interface**: Streamlit-based user interface for real-time predictions
- **Comprehensive Evaluation**: Multiple metrics including precision, recall, F1-score, and accuracy

## 📊 Dataset

- **Source**: [Wound Classification Dataset](https://www.kaggle.com/datasets/ibrahimfateen/wound-classification) by Ibrahim Fateen
- **License**: CC0 Public Domain
- **Images**: 442 total images (242 bruise images, 200 normal images)
- **Formats**: JPG, JPEG, PNG
- **Last Updated**: 2022

### Data Distribution
- **Training**: 70%
- **Validation**: 20% 
- **Testing**: 10%

## 🛠️ Technical Stack

### Core Technologies
- **Python**: 3.11
- **Deep Learning**: TensorFlow 2.x
- **Development Environment**: Visual Studio Code with Jupyter notebooks
- **Web Framework**: Streamlit

### Libraries
- **NumPy**: Numerical computations
- **Pillow**: Image processing
- **Scikit-learn**: Dataset management and metrics
- **Matplotlib & Seaborn**: Data visualization
- **TensorFlow/Keras**: Deep learning framework

## 🔄 Data Preprocessing Pipeline

1. **Image Resizing**: Standardized to 224×224 pixels
2. **Color Space Normalization**: RGB channel normalization
3. **Pixel Value Normalization**: Values scaled to [0,1] range
4. **Error Handling**: Robust processing for corrupt images
5. **Format Validation**: Support for JPEG, PNG, JPG formats

## 📈 Experimental Phases

### Phase 1: Baseline Model
- Development of fundamental CNN architecture
- Performance baseline establishment
- Basic training parameter optimization

### Phase 2: Experimental Comparison
- **Data Augmentation**: Horizontal flipping, rotation (±20°), zooming (±20%)
- **Architecture Optimization**: Comparison of shallow vs. deep networks
- **Training Enhancement**: Optimizer comparison (Adam vs. SGD)
- **Regularization Experimentation**: Dropout rate optimization and L2 regularization

### Phase 3: Integration & Deployment
- Model optimization integration
- Streamlit web interface development
- Performance validation and testing

## 📊 Evaluation Metrics

The model performance is evaluated using multiple metrics:

- **Precision**: `TP / (TP + FP)`
- **Recall (Sensitivity)**: `TP / (TP + FN)`
- **F1-Score**: `2 * (Precision * Recall) / (Precision + Recall)`
- **Accuracy**: `(TP + TN) / Total Samples`

Additional visualizations include:
- ROC Curves
- Precision-Recall Curves  
- Confusion Matrices

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.11+
TensorFlow 2.x
NumPy
Pillow
Scikit-learn
Matplotlib
Seaborn
Streamlit
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/bruise-detection-cnn.git
cd bruise-detection-cnn

# Install dependencies
pip install -r requirements.txt

# Run the download python file. Download the dataset
# Place dataset in data/ directory following the structure:
# data/
#   ├── bruise/
#   └── normal/
```

### Usage

#### Running the Web Interface
```bash
streamlit run app.py
```

## 📁 Project Structure
```
bruise-detection-cnn/
├── data/                   # Dataset directory
├── models/                 # Trained model files
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Source code
│   ├── preprocessing.py    # Data preprocessing utilities
│   ├── model.py           # CNN model architecture
│   ├── train.py           # Training scripts
│   └── evaluate.py        # Evaluation utilities
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

