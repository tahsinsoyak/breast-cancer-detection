# Breast Cancer Classification with Neural Networks: Advanced Model Evaluation Framework

This project implements a comprehensive breast cancer classification system using neural networks with multiple evaluation methodologies. The system analyzes breast cancer data to classify tumors as either benign or malignant, providing medical professionals with a reliable diagnostic tool.

The framework features three distinct evaluation approaches: 5-fold cross-validation, 10-fold cross-validation, and random split validation. Each method employs different neural network architectures with configurable parameters to ensure robust model evaluation. The system includes advanced visualization capabilities for model architecture and performance metrics, making it particularly valuable for both research and clinical applications.

## Repository Structure
```
.
├── 10-Fold/
│   └── 10-Fold.py          # Implementation of 10-fold cross-validation evaluation
├── 5-Fold/
│   └── 5-Fold.py          # Implementation of 5-fold cross-validation evaluation
├── Rastgele Ayırma/
│   └── Rastgele_Ayirma.py # Random split evaluation implementation
├── requirements.txt       # Project dependencies and versions
└── test.py              # Main test script with core model implementation
```

## Usage Instructions
### Prerequisites
- Python 3.x
- TensorFlow 2.18.0
- Scikit-learn 1.5.2
- Pandas 2.2.3
- NumPy 2.0.2
- Matplotlib 3.9.3
- Seaborn 0.13.2

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install required packages
pip install -r requirements.txt
```

### Quick Start
1. Prepare your data:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
df = pd.read_csv('breast_cancer_data.csv')
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis'].map({'B': 1, 'M': 0})

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

2. Run the desired evaluation method:
```python
# For 5-fold cross-validation
python 5-Fold/5-Fold.py

# For 10-fold cross-validation
python 10-Fold/10-Fold.py

# For random split evaluation
python "Rastgele Ayırma/Rastgele_Ayirma.py"
```

### More Detailed Examples
1. Custom model configuration:
```python
# Define custom model parameters
model_config = {
    'name': 'Custom Model',
    'hidden_layer_sizes': (128, 64),
    'activation': 'relu',
    'optimizer': 'adam'
}

# Create and train model
model = create_model(
    hidden_layer_sizes=model_config['hidden_layer_sizes'],
    activation=model_config['activation'],
    optimizer=model_config['optimizer']
)
```

2. Visualizing results:
```python
# Plot confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

### Troubleshooting
1. Memory Issues
- **Problem**: `OutOfMemoryError` during model training
- **Solution**: Reduce batch size in model training
```python
model.fit(X_train, y_train, batch_size=16, epochs=50)  # Reduced from 32
```

2. Poor Model Performance
- **Problem**: Low accuracy or high variance
- **Solution**: Adjust model architecture or hyperparameters
```python
# Increase model complexity
model_config = {
    'hidden_layer_sizes': (256, 128),
    'activation': 'relu',
    'optimizer': 'adam'
}
```

3. Data Preprocessing Issues
- **Problem**: `ValueError` with feature scaling
- **Solution**: Check for missing values and handle them
```python
# Handle missing values
df = df.dropna()
# Then proceed with scaling
X_scaled = scaler.fit_transform(X)
```

## Data Flow
The system processes breast cancer data through a pipeline of preprocessing, model training, and evaluation stages.

```ascii
Raw Data → Preprocessing → Feature Scaling → Model Training → Evaluation
[CSV Input] → [Clean+Format] → [StandardScaler] → [Neural Network] → [Metrics]
```

Key component interactions:
1. Data Loader loads and validates the breast cancer dataset
2. Preprocessor removes unnecessary columns and handles missing values
3. StandardScaler normalizes feature values to improve model training
4. Neural Network processes scaled features through configurable layers
5. Evaluation Module performs cross-validation and generates performance metrics
6. Visualization Component generates confusion matrices and performance plots
7. Model Saver stores trained models for future use