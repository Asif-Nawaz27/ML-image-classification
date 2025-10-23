# Advanced Image Classification Techniques in Computer Vision

This project implements machine learning models for multi-label medical image classification. Specifically, it uses a Convolutional Neural Network (CNN), VGG, RESTNET and INCEPTION model to classify chest X-ray images across multiple pathological conditions simultaneously. The following section explains the CNN model for better understanding of the code

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Prerequisites and Dependencies](#prerequisites-and-dependencies)
- [Project Structure](#project-structure)
- [Implementation Steps](#implementation-steps)
  - [1. Environment Setup and Imports](#1-environment-setup-and-imports)
  - [2. Configuration Variables](#2-configuration-variables)
  - [3. Utility Functions](#3-utility-functions)
  - [4. Data Loading and Preprocessing](#4-data-loading-and-preprocessing)
  - [5. Data Augmentation](#5-data-augmentation)
  - [6. Data Visualization](#6-data-visualization)
  - [7. Class Frequency Analysis](#7-class-frequency-analysis)
  - [8. Model Architecture](#8-model-architecture)
  - [9. Hyperparameter Optimization](#9-hyperparameter-optimization)
  - [10. Model Training](#10-model-training)
  - [11. Model Evaluation](#11-model-evaluation)
  - [12. Results Visualization](#12-results-visualization)
- [Model Performance](#model-performance)
- [Usage Instructions](#usage-instructions)
- [Results and Analysis](#results-and-analysis)

## Project Overview

This project focuses on developing a CNN-based deep learning model for multi-label classification of medical images, specifically chest X-rays. The model can identify multiple pathological conditions simultaneously, making it suitable for real-world medical diagnosis applications.

**Key Features:**
- Multi-label classification capability
- Data augmentation for improved generalization
- Hyperparameter optimization using grid search
- Comprehensive evaluation with multiple metrics
- ROC curve analysis for each condition
- Confusion matrix visualization

## Dataset Information

The project uses a medical imaging dataset containing chest X-ray images with multiple pathological labels. The dataset is structured with:
- **Image files**: Chest X-ray images in various formats
- **Labels**: Multiple binary labels for different pathological conditions
- **Patient metadata**: Patient ID and other relevant information

## Prerequisites and Dependencies

```python
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Machine Learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from tensorflow.keras.callbacks import LearningRateScheduler

# Scikit-learn libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, classification_report, multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Additional utilities
from scikeras.wrappers import KerasClassifier
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings('ignore')
```

## Project Structure

```
image-classification/
├── Models/CNN_V1.ipynb              # Main implementation notebook
├── data/                     # Dataset directory
│   ├── images/              # Image files
│   └── labels.csv           # Labels and metadata
├── models/                   # Saved models
├── results/                  # Output visualizations
└── README.md                # This documentation
```

## Implementation Steps

### 1. Environment Setup and Imports

The project begins by importing all necessary libraries for data manipulation, machine learning, and visualization:

```python
# Set random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
```

**Key Libraries Used:**
- **TensorFlow/Keras**: Deep learning framework for CNN implementation
- **Scikit-learn**: Machine learning utilities and evaluation metrics
- **Pandas/NumPy**: Data manipulation and numerical operations
- **Matplotlib/Seaborn**: Data visualization and plotting

### 2. Configuration Variables

Global configuration parameters are defined to ensure consistency across the project:

```python
# Model configuration
IMAGE_SIZE = [128, 128]    # Input image dimensions
EPOCHS = 20                # Number of training epochs
BATCH_SIZE = 32           # Batch size for training
RANDOM_STATE = 3          # Random state for reproducibility
SEED = 42                 # Global seed value

# Define target labels for multi-label classification
labels = ['Cardiomegaly', 'Emphysema', 'Effusion', ...]  # Specific to your dataset
```

**Configuration Details:**
- **IMAGE_SIZE**: Standardized input dimensions for consistent processing
- **EPOCHS**: Number of complete passes through the training data
- **BATCH_SIZE**: Number of samples processed before model update
- **Labels**: List of all possible pathological conditions

### 3. Utility Functions

Two critical utility functions are implemented for model evaluation:

#### Training Visualization Function
```python
def visualize_training(history):
    """
    Visualizes training and validation metrics over epochs
    
    Parameters:
    history: Training history object from model.fit()
    """
    # Create subplots for loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Plot training & validation accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

#### ROC Curve Analysis Function
```python
def get_roc_curve(labels, y_true, y_score, when=''):
    """
    Generates ROC curves for multi-label classification
    
    Parameters:
    labels: List of label names
    y_true: True labels (binary matrix)
    y_score: Prediction scores
    when: Description of when this evaluation occurs
    
    Returns:
    auc_scores: Dictionary of AUC scores for each label
    """
    auc_scores = {}
    
    # Create subplots for ROC curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, label in enumerate(labels):
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        auc_score = roc_auc_score(y_true[:, i], y_score[:, i])
        auc_scores[label] = auc_score
        
        # Plot ROC curve
        axes[i].plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--')
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'ROC Curve - {label} ({when})')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return auc_scores
```

### 4. Data Loading and Preprocessing

The data loading process involves several critical steps:

#### Dataset Preparation
```python
# Load main dataset
train_df_main = pd.read_csv('path/to/labels.csv')

# Create image path mapping
image_paths = glob.glob('path/to/images/*')
index_path_map = pd.DataFrame({
    'Image Index': [os.path.basename(path) for path in image_paths],
    'FilePath': image_paths
})

# Split data into train/validation/test sets
train_and_valid_set, test_set = train_test_split(
    train_df_main, 
    test_size=0.2, 
    random_state=RANDOM_STATE,
    stratify=train_df_main[labels].values
)

train_set, valid_set = train_test_split(
    train_and_valid_set, 
    test_size=0.25, 
    random_state=RANDOM_STATE,
    stratify=train_and_valid_set[labels].values
)
```

#### Data Merging and Cleaning
```python
# Merge datasets with file paths
train_set = pd.merge(train_set, index_path_map, on='Image Index', how='left')
valid_set = pd.merge(valid_set, index_path_map, on='Image Index', how='left')
test_set = pd.merge(test_set, index_path_map, on='Image Index', how='left')

# Remove duplicates
train_set = train_set.drop_duplicates()
valid_set = valid_set.drop_duplicates()
test_set = test_set.drop_duplicates()

# Print dataset statistics
print(f'Main dataframe: {train_df_main.shape}')
print(f'Training set: {train_set.shape}')
print(f'Validation set: {valid_set.shape}')
print(f'Test set: {test_set.shape}')
```

**Key Steps:**
1. **Data Loading**: Read CSV files containing image metadata and labels
2. **Path Mapping**: Create mapping between image indices and file paths
3. **Data Splitting**: Stratified split to maintain label distribution
4. **Data Merging**: Combine labels with corresponding image paths
5. **Duplicate Removal**: Ensure data quality by removing duplicates

### 5. Data Augmentation

Data augmentation is crucial for improving model generalization and handling limited training data:

#### Training Data Generator
```python
def get_train_generator(df, image_dir, x_col, y_cols, sample_weight_col, 
                       shuffle=True, batch_size=8, seed=1, 
                       target_w=320, target_h=320):
    """
    Creates augmented training data generator
    
    Parameters:
    df: Training dataframe
    image_dir: Directory containing images
    x_col: Column name for image paths
    y_cols: Column names for labels
    batch_size: Number of samples per batch
    target_w, target_h: Target image dimensions
    
    Returns:
    generator: Keras data generator with augmentation
    """
    
    # Define augmentation parameters
    image_generator = ImageDataGenerator(
        samplewise_center=True,              # Center each sample
        samplewise_std_normalization=True,   # Normalize each sample
        shear_range=0.1,                     # Shear transformation
        zoom_range=0.15,                     # Random zoom
        rotation_range=32,                   # Random rotation (±32°)
        width_shift_range=0.1,               # Horizontal shift
        height_shift_range=0.05,             # Vertical shift
        horizontal_flip=True,                # Random horizontal flip
        vertical_flip=False,                 # No vertical flip
        brightness_range=(0.8, 1.2),        # Brightness variation
        channel_shift_range=0.1,             # Color channel shift
        rescale=1./255,                      # Normalize pixel values
        fill_mode='reflect'                  # Fill mode for transformations
    )
    
    # Create data generator
    generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        target_size=(target_w, target_h)
    )
    
    return generator
```

#### Validation and Test Data Generator
```python
def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, 
                                x_col, y_cols, sample_weight_col,
                                sample_size=100, batch_size=8, seed=1, 
                                target_w=320, target_h=320):
    """
    Creates validation and test data generators without augmentation
    
    Key differences from training generator:
    - No augmentation applied
    - Uses statistics from training data for normalization
    - Fixed order (shuffle=False) for consistent evaluation
    """
    
    # Sample training data for normalization statistics
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col="FilePath",
        y_col=labels,
        class_mode="raw",
        batch_size=sample_size,
        shuffle=True,
        target_size=(target_w, target_h)
    )
    
    # Get sample batch for fitting normalization
    batch = raw_train_generator.__next__()
    data_sample = batch[0]
    
    # Create generator with training data statistics
    image_generator = ImageDataGenerator(
        featurewise_center=True,              # Use training mean
        featurewise_std_normalization=True,   # Use training std
        rescale=1./255
    )
    
    # Fit generator to training data sample
    image_generator.fit(data_sample)
    
    # Create validation and test generators
    valid_generator = image_generator.flow_from_dataframe(
        dataframe=valid_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,  # Important: no shuffling for evaluation
        seed=seed,
        target_size=(target_w, target_h)
    )
    
    test_generator = image_generator.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,  # Important: no shuffling for evaluation
        seed=seed,
        target_size=(target_w, target_h)
    )
    
    return valid_generator, test_generator
```

**Augmentation Techniques Used:**
- **Geometric Transformations**: Rotation, shearing, zooming, shifting
- **Photometric Transformations**: Brightness adjustment, channel shifting
- **Flipping**: Horizontal flip (medical images typically don't require vertical flip)
- **Normalization**: Pixel value normalization and standardization

### 6. Data Visualization

Visualization helps understand the dataset and verify data loading:

#### Sample Images Display
```python
def get_label(y):
    """Convert binary label array to readable string"""
    ret_labels = []
    for i, idx in enumerate(y):
        if idx:
            ret_labels.append(labels[i])
    return '|'.join(ret_labels) if ret_labels else 'No Label'

# Get batch of images for visualization
x, y = train_generator.__getitem__(0)

# Display sample images with labels
fig, axs = plt.subplots(2, 6, figsize=(20, 10))
for i in range(12):
    axs[i//6, i%6].imshow(x[i])
    axs[i//6, i%6].set_title(get_label(y[i]))
    axs[i//6, i%6].axis('on')

plt.tight_layout()
plt.show()
```

#### Label Distribution Analysis
```python
# Plot label distribution in training set
plt.figure(figsize=(8, 4))
plt.xticks(rotation=90)
plt.bar(labels, train_generator.labels.sum(axis=0)/train_generator.n * 100)
plt.title('Percentage of different conditions in train dataset')
plt.xlabel('Conditions')
plt.ylabel('Percentage')
plt.show()
```

### 7. Class Frequency Analysis

Understanding class imbalance is crucial for multi-label medical classification:

```python
def compute_class_freqs(labels):
    """
    Calculate positive and negative class frequencies
    
    Parameters:
    labels: Binary label matrix
    
    Returns:
    positive_frequencies: Frequency of positive cases
    negative_frequencies: Frequency of negative cases
    """
    N = labels.shape[0]
    positive_frequencies = labels.sum(axis=0) / N
    negative_frequencies = 1.0 - positive_frequencies
    return positive_frequencies, negative_frequencies

# Calculate class frequencies
freq_pos, freq_neg = compute_class_freqs(train_generator.labels)

# Visualize class imbalance
data_pos = pd.DataFrame({
    "Class": labels, 
    "Label": "Positive", 
    "Value": freq_pos
})
data_neg = pd.DataFrame({
    "Class": labels, 
    "Label": "Negative", 
    "Value": freq_neg
})

data = pd.concat([data_pos, data_neg], ignore_index=True)
plt.figure(figsize=(12, 6))
sns.barplot(x="Class", y="Value", hue="Label", data=data)
plt.xticks(rotation=90)
plt.title('Class Distribution: Positive vs Negative Cases')
plt.show()
```

### 8. Model Architecture

The CNN architecture is designed specifically for multi-label medical image classification:

```python
def cnn_model(optimizer='Adam', init_mode='uniform', activation='relu', 
              dropout_rate=0.5, connected_activation='sigmoid', 
              learn_rate=0.01, isCompile=False):
    """
    Creates CNN model for multi-label classification
    
    Architecture:
    - Input layer: Accepts images of size IMAGE_SIZE + 3 channels
    - Convolutional blocks: 3 blocks with increasing filter sizes
    - Each block: Conv2D → Conv2D → MaxPooling → Dropout
    - Dense layers: Fully connected layer + output layer
    - Output: Sigmoid activation for multi-label classification
    
    Parameters:
    optimizer: Optimization algorithm
    init_mode: Weight initialization method
    activation: Activation function for hidden layers
    dropout_rate: Dropout rate for regularization
    connected_activation: Activation for output layer
    learn_rate: Learning rate
    isCompile: Whether to compile the model
    
    Returns:
    model: Compiled Keras model
    """
    
    model = Sequential()
    
    # Input Layer
    model.add(Input(shape=(*IMAGE_SIZE, 3)))
    model.add(Conv2D(64, (3, 3), padding='same', 
                     activation=activation, kernel_initializer=init_mode))
    
    # Convolutional Block 1
    model.add(Conv2D(32, (3, 3), activation=activation, 
                     kernel_initializer=init_mode))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))
    
    # Convolutional Block 2
    model.add(Conv2D(64, (3, 3), padding='same', 
                     activation=activation, kernel_initializer=init_mode))
    model.add(Conv2D(64, (3, 3), activation=activation, 
                     kernel_initializer=init_mode))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))
    
    # Convolutional Block 3
    model.add(Conv2D(128, (3, 3), padding='same', 
                     activation=activation, kernel_initializer=init_mode))
    model.add(Conv2D(128, (3, 3), activation=activation, 
                     kernel_initializer=init_mode))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))
    
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(1024, activation=activation, 
                    kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    
    # Output Layer (Multi-label classification)
    model.add(Dense(len(labels), activation=connected_activation))
    
    # Get optimizer class
    optimizer_class = get_optimizer(optimizer)
    
    # Compile model
    if isCompile:
        model.compile(
            optimizer=optimizer_class(learning_rate=learn_rate),
            loss='binary_crossentropy',  # For multi-label classification
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer=optimizer_class(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return model
```

**Architecture Details:**
- **Input Layer**: Accepts RGB images of specified dimensions
- **Convolutional Blocks**: Three progressive blocks with increasing complexity
  - Block 1: 32 filters (3×3), focuses on basic features
  - Block 2: 64 filters (3×3), captures mid-level features  
  - Block 3: 128 filters (3×3), extracts high-level features
- **Pooling**: MaxPooling2D reduces spatial dimensions
- **Regularization**: Dropout layers prevent overfitting
- **Dense Layers**: 1024-unit fully connected layer for feature integration
- **Output**: Sigmoid activation for independent binary classifications

### 9. Hyperparameter Optimization

The project implements comprehensive hyperparameter tuning using grid search:

#### Batch Size and Epochs Optimization
```python
def get_batch_parameter(search_model):
    """
    Searches for optimal batch size and number of epochs
    
    Parameters:
    search_model: Model function to optimize
    
    Returns:
    best_params: Dictionary with optimal batch_size and epochs
    """
    tf.random.set_seed(SEED)
    
    # Create KerasClassifier wrapper
    model = KerasClassifier(model=search_model, verbose=0)
    
    # Define parameter grid
    batch_size = [10, 20, 40, 60, 80]
    epochs = [10, 15, 20]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    
    # Perform randomized search
    grid = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid, 
        n_iter=12, 
        n_jobs=-1, 
        cv=5
    )
    
    # Fit on sample data
    batches_x, batches_y = train_generator.__getitem__(0)
    grid_result = grid.fit(batches_x, batches_y)
    
    print(f"Best: {grid_result.best_score_:.4f} using {grid_result.best_params_}")
    return grid_result.best_params_
```

#### Comprehensive Parameter Search
```python
def get_parameters_using_batch(search_model, batch=10, epoch=10):
    """
    Searches for optimal model hyperparameters
    
    Parameters tested:
    - Optimizer: SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
    - Learning rate: Various values from 0.00001 to 0.3
    - Weight initialization: Multiple initialization strategies
    - Activation functions: Different activation functions
    - Dropout rate: Values from 0.0 to 0.9
    
    Returns:
    best_params: Dictionary with optimal hyperparameters
    """
    tf.random.set_seed(SEED)
    
    model = KerasClassifier(
        model=search_model, 
        loss="binary_crossentropy", 
        epochs=epoch, 
        batch_size=batch, 
        verbose=0
    )
    
    # Define comprehensive parameter grid
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 
                 'Adam', 'Adamax', 'Nadam']
    learn_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 
                 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    activation = ['softmax', 'softplus', 'softsign', 'relu', 
                  'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    param_grid = dict(
        optimizer=optimizer, 
        optimizer__learning_rate=learn_rate,
        model__init_mode=init_mode, 
        model__activation=activation, 
        model__dropout_rate=dropout_rate
    )
    
    # Perform randomized search
    grid = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid, 
        n_iter=15, 
        n_jobs=-1, 
        cv=5
    )
    
    x, y = train_generator.__getitem__(0)
    grid_result = grid.fit(x, y)
    
    print(f"Best: {grid_result.best_score_:.4f} using {grid_result.best_params_}")
    return grid_result.best_params_
```

#### Learning Rate Scheduling
```python
def build_lrfn(lr_start=0.000002, lr_max=0.00010, lr_min=0, 
               lr_rampup_epochs=8, lr_sustain_epochs=0, lr_exp_decay=0.8):
    """
    Creates learning rate schedule function
    
    Learning rate strategy:
    1. Ramp-up phase: Gradually increase LR
    2. Sustain phase: Maintain maximum LR
    3. Decay phase: Exponentially decay LR
    
    Parameters:
    lr_start: Initial learning rate
    lr_max: Maximum learning rate
    lr_min: Minimum learning rate
    lr_rampup_epochs: Epochs for ramp-up
    lr_sustain_epochs: Epochs to sustain max LR
    lr_exp_decay: Exponential decay factor
    
    Returns:
    lrfn: Learning rate function
    """
    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            # Ramp-up phase
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            # Sustain phase
            lr = lr_max
        else:
            # Decay phase
            lr = (lr_max - lr_min) * \
                 lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    return lrfn

# Create learning rate scheduler
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
```

### 10. Model Training

The training process involves finding optimal parameters and training the final model:

```python
# Step 1: Find optimal batch size and epochs
cnn_best_batch_param = get_batch_parameter(cnn_model)
cnn_batch_size = cnn_best_batch_param['batch_size']
cnn_epoch = cnn_best_batch_param['epochs']

# Step 2: Find optimal hyperparameters
cnn_best_param = get_parameters_using_batch(
    search_model=cnn_model, 
    batch=cnn_batch_size, 
    epoch=cnn_epoch
)

# Step 3: Extract best parameters
cnn_optimizer = cnn_best_param['optimizer']
cnn_init_mode = cnn_best_param['model__init_mode']
cnn_activation = cnn_best_param['model__activation']
cnn_dropout_rate = cnn_best_param['model__dropout_rate']
cnn_connected_activation = 'sigmoid'
cnn_learning_rate = cnn_best_param['optimizer__learning_rate']

# Step 4: Create final model with optimal parameters
model = cnn_model(
    optimizer=cnn_optimizer,
    init_mode=cnn_init_mode,
    activation=cnn_activation,
    dropout_rate=cnn_dropout_rate,
    connected_activation=cnn_connected_activation,
    learn_rate=cnn_learning_rate,
    isCompile=True
)

# Display model architecture
model.summary()

# Step 5: Baseline predictions (before training)
predicted_vals_before = model.predict(test_generator, steps=len(test_generator))

# Step 6: Train the model
history = model.fit(
    train_generator,
    epochs=cnn_epoch,
    validation_data=valid_generator,
    batch_size=cnn_batch_size,
    verbose=1,
    callbacks=[lr_schedule]  # Apply learning rate scheduling
)

# Step 7: Visualize training progress
visualize_training(history)
```

**Training Process:**
1. **Parameter Optimization**: Systematic search for optimal hyperparameters
2. **Model Creation**: Build model with best-found parameters
3. **Baseline Evaluation**: Record performance before training
4. **Training**: Fit model with learning rate scheduling
5. **Monitoring**: Track training and validation metrics

### 11. Model Evaluation

Comprehensive evaluation using multiple metrics and visualizations:

#### Basic Performance Metrics
```python
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(
    test_generator, 
    steps=test_generator.samples // test_generator.batch_size, 
    verbose=1
)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Generate predictions
predicted_vals_after = model.predict(test_generator, steps=len(test_generator))

# ROC analysis
auc_rocs_before = get_roc_curve(
    labels, 
    test_generator.labels, 
    predicted_vals_before, 
    when='before training'
)

auc_rocs_after = get_roc_curve(
    labels, 
    test_generator.labels, 
    predicted_vals_after, 
    when='after training'
)
```

#### Detailed Classification Metrics
```python
# Convert predictions to class labels
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_generator.labels, axis=1)

# Calculate comprehensive metrics
f1 = f1_score(true_labels, predicted_labels, average='macro')
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
report = classification_report(true_labels, predicted_labels, target_names=labels)

# Display results
print(f'Accuracy: {test_accuracy:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print('\nClassification Report:')
print(report)
```

### 12. Results Visualization

#### Confusion Matrix Visualization
```python
# Generate confusion matrices for all labels
confusion_mtx = multilabel_confusion_matrix(true_labels, predicted_labels)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 30))
gs = gridspec.GridSpec(
    nrows=(len(labels) + 1) // 2, 
    ncols=3, 
    height_ratios=[1]*((len(labels) + 1) // 2), 
    hspace=0.5
)

for i in range(len(labels)):
    ax = fig.add_subplot(gs[i])
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx[i])
    disp.plot(cmap="Blues", ax=ax)
    
    # Customize appearance
    ax.set_title(f'Confusion Matrix for {labels[i]}', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xticklabels(['False', 'True'], rotation=90, fontsize=10)
    ax.set_yticklabels(['False', 'True'], rotation=0, fontsize=10)

plt.tight_layout()
plt.show()
```

## Model Performance

The model's performance is evaluated using multiple metrics:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Proportion of positive identifications that were correct
- **Recall**: Proportion of actual positives that were identified correctly
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve for each label
- **Confusion Matrix**: Detailed breakdown of classification results per label

## Usage Instructions

1. **Setup Environment**:
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn scikit-learn tqdm
   ```

2. **Prepare Data**:
   - Place image files in the designated directory
   - Ensure CSV file contains proper labels and image references
   - Update file paths in the configuration section

3. **Run the Notebook**:
   - Execute cells sequentially
   - Monitor training progress and adjust parameters if needed
   - Review evaluation results and visualizations

4. **Model Customization**:
   - Modify `IMAGE_SIZE` for different input dimensions
   - Adjust `labels` list for your specific classification task
   - Tune hyperparameters in the optimization functions

## Results and Analysis

The project provides comprehensive analysis including:

- **Training Dynamics**: Loss and accuracy curves showing model convergence
- **Class Distribution**: Understanding of dataset imbalance
- **ROC Curves**: Performance analysis for each individual label
- **Confusion Matrices**: Detailed classification results
- **Comparative Analysis**: Before and after training performance

The multi-label CNN architecture demonstrates effective learning capabilities for medical image classification, with proper handling of class imbalance and comprehensive evaluation metrics providing insights into model performance across all target conditions.

---

*This implementation serves as a complete framework for multi-label medical image classification using CNNs, with extensive documentation and evaluation capabilities.*
