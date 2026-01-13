# Project Code Explanation

This document provides a detailed explanation of each file in the project, what it does, and how the pieces fit together. Use this to understand the codebase and explain it to others.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [File-by-File Explanation](#file-by-file-explanation)
   - [config.py](#configpy---configuration)
   - [data_loader.py](#data_loaderpy---data-loading-and-augmentation)
   - [model.py](#modelpy---neural-network-architectures)
   - [train.py](#trainpy---training-script)
   - [evaluate.py](#evaluatepy---model-evaluation)
   - [realtime_detector.py](#realtime_detectorpy---webcam-detection)
   - [utils.py](#utilspy---helper-functions)
3. [How the Pipeline Works](#how-the-pipeline-works)
4. [Key Concepts Explained](#key-concepts-explained)
5. [Files NOT Used (Legacy)](#files-not-used-legacy)

---

## Project Overview

The project follows a modular architecture where each file has a single responsibility:

```
User runs train.py
        |
        v
config.py (loads settings) --> data_loader.py (loads images) --> model.py (creates CNN)
        |
        v
Training loop with callbacks (EarlyStopping, ModelCheckpoint)
        |
        v
Saved model: models/emotion_cnn_best.h5
        |
        v
evaluate.py (generates metrics) OR realtime_detector.py (live detection)
```

---

## File-by-File Explanation

### config.py - Configuration

**Purpose**: Single source of truth for all settings, paths, and hyperparameters.

**Key Components**:

1. **GPU Setup Function**:
```python
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
```
This prevents TensorFlow from grabbing all GPU memory at once. Instead, it allocates memory as needed.

2. **Paths**:
```python
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'fer2013 dataset', 'train')
TEST_DIR = os.path.join(PROJECT_ROOT, 'fer2013 dataset', 'test')
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'emotion_cnn_best.h5')
```
All file paths are defined here so they can be changed in one place.

3. **Emotion Labels**:
```python
EMOTION_LABELS = {
    0: 'angry',
    1: 'disgust', 
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}
```
This mapping is used everywhere in the project. The order matches alphabetical folder sorting.

4. **Class Weights**:
```python
CLASS_WEIGHTS = {
    0: 1.8,    # angry
    1: 16.5,   # disgust (severely underrepresented)
    2: 1.8,    # fear
    3: 1.0,    # happy (baseline - most samples)
    ...
}
```
These weights tell the model to penalize mistakes on rare classes more heavily. Disgust has weight 16.5 because it has 16x fewer samples than Happy.

5. **Hyperparameters**:
```python
BATCH_SIZE = 64
EPOCHS = 50
INITIAL_LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
```

**Why it matters**: Centralizing configuration prevents "magic numbers" scattered throughout the code and makes experiments reproducible.

---

### data_loader.py - Data Loading and Augmentation

**Purpose**: Load images from folders, apply augmentation, and create train/validation/test splits.

**Key Components**:

1. **ImageDataGenerator for Training**:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixels to [0,1]
    rotation_range=15,        # Random rotation up to 15 degrees
    width_shift_range=0.1,    # Shift image horizontally up to 10%
    height_shift_range=0.1,   # Shift image vertically up to 10%
    horizontal_flip=True,     # Randomly flip images
    zoom_range=0.1,           # Random zoom
    brightness_range=[0.8, 1.2],  # Vary brightness
    validation_split=0.2      # Reserve 20% for validation
)
```

2. **ImageDataGenerator for Test**:
```python
test_datagen = ImageDataGenerator(rescale=1./255)  # Only normalize, no augmentation
```
Test data should NOT be augmented - we want to evaluate on real, unmodified images.

3. **flow_from_directory**:
```python
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=64,
    subset='training'
)
```
This automatically loads images from folders, where folder names become class labels.

**Why augmentation matters**: With only ~29,000 training images, the model could easily memorize them. Augmentation creates variations (rotated, shifted, flipped) so the model learns general patterns rather than specific images.

---

### model.py - Neural Network Architectures

**Purpose**: Define the CNN architecture and an optional EfficientNet transfer learning model.

**The CNN Architecture**:

```python
model = models.Sequential([
    # Block 1: 64 filters
    layers.Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Block 2: 128 filters (spatial: 24x24)
    # Block 3: 256 filters (spatial: 12x12)
    # Block 4: 512 filters (spatial: 6x6)
    
    # Classifier
    layers.GlobalAveragePooling2D(),
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])
```

**Layer-by-Layer Explanation**:

1. **Conv2D**: Convolutional layer that learns filters to detect features (edges, textures, shapes)
2. **BatchNormalization**: Normalizes layer outputs, speeds up training, adds regularization
3. **ReLU Activation**: Introduces non-linearity: `f(x) = max(0, x)`
4. **MaxPooling2D**: Reduces spatial dimensions by taking max in 2x2 windows
5. **Dropout**: Randomly sets 25% (or 50%) of neurons to zero during training to prevent overfitting
6. **GlobalAveragePooling2D**: Averages each feature map to a single value (reduces 6x6x512 to 512)
7. **Dense**: Fully connected layer for classification
8. **Softmax**: Converts outputs to probabilities that sum to 1

**Why 4 blocks?**: Each block doubles the filters (64->128->256->512) while halving spatial dimensions (48->24->12->6->3). This creates a pyramid that captures both low-level features (edges) and high-level features (facial expressions).

---

### train.py - Training Script

**Purpose**: Execute the training loop with proper callbacks and logging.

**Key Components**:

1. **Callbacks**:
```python
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5
    ),
    ModelCheckpoint(
        filepath='models/emotion_cnn_best.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    TensorBoard(log_dir='logs/cnn_TIMESTAMP')
]
```

- **EarlyStopping**: Stops training if validation accuracy doesn't improve for 10 epochs
- **ReduceLROnPlateau**: Halves learning rate if stuck for 5 epochs
- **ModelCheckpoint**: Saves the best model (highest val_accuracy) automatically
- **TensorBoard**: Logs training metrics for visualization

2. **Training Loop**:
```python
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    class_weight=CLASS_WEIGHTS,
    callbacks=callbacks
)
```

**Why class_weight?**: Without it, the model would learn to always predict "Happy" (most common) and achieve 25% accuracy. Class weights force it to care equally about all emotions.

---

### evaluate.py - Model Evaluation

**Purpose**: Generate comprehensive metrics and visualizations for the trained model.

**Key Outputs**:

1. **Classification Report**:
```
              precision    recall  f1-score   support
       angry     0.56      0.63      0.59       958
     disgust     0.63      0.59      0.61       111
        fear     0.50      0.46      0.48      1024
       happy     0.90      0.84      0.87      1774
     neutral     0.59      0.68      0.63      1233
         sad     0.57      0.50      0.53      1247
    surprise     0.72      0.78      0.75       831
```

- **Precision**: Of all images predicted as X, what % were actually X?
- **Recall**: Of all actual X images, what % did we correctly identify?
- **F1-Score**: Harmonic mean of precision and recall

2. **Confusion Matrix**:
Shows which emotions are confused with each other. For example:
- Fear often predicted as Sad (183 samples)
- Sad often predicted as Neutral (237 samples)

3. **Most Confused Pairs**:
```
sad -> neutral: 237 samples (19.0%)
fear -> sad: 183 samples (17.9%)
```

---

### realtime_detector.py - Webcam Detection

**Purpose**: Use the trained model for live webcam-based emotion detection.

**Key Components**:

1. **Face Detection**:
```python
self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
faces = self.face_cascade.detectMultiScale(
    gray_frame,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(30, 30)
)
```
Uses OpenCV's Haar Cascade to find face bounding boxes in each frame.

2. **Preprocessing**:
```python
def preprocess_face(self, face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=-1)  # Add channel dim
    face_img = np.expand_dims(face_img, axis=0)   # Add batch dim
    return face_img
```
Converts webcam frame to same format as training data.

3. **Temporal Smoothing**:
```python
self.prediction_history = deque(maxlen=5)
self.prediction_history.append(probs)
smoothed_probs = np.mean(self.prediction_history, axis=0)
```
Averages predictions over last 5 frames to reduce jitter. Without this, the prediction would flicker rapidly between emotions.

4. **Main Loop**:
```python
while True:
    ret, frame = cap.read()
    faces = self.detect_faces(frame)
    for face in faces:
        emotion, confidence, probs = self.predict(face)
        self.draw_results(frame, emotion, confidence)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

### utils.py - Helper Functions

**Purpose**: Utility functions used by other scripts.

**Contents**:

1. **plot_training_history**: Plots accuracy/loss curves from training
2. **plot_confusion_matrix**: Creates confusion matrix visualization
3. **get_class_distribution**: Counts samples per emotion class

---

## How the Pipeline Works

### Training Pipeline

```
1. config.py loads all settings
2. data_loader.py creates generators that:
   - Load images from fer2013 dataset/train/
   - Apply augmentation (rotate, flip, shift, etc.)
   - Split into train (80%) and validation (20%)
3. model.py builds the CNN architecture
4. train.py runs the training loop:
   - Forward pass: images -> CNN -> predictions
   - Loss calculation: compare predictions to true labels
   - Backward pass: calculate gradients
   - Update weights: optimizer adjusts model
   - Repeat for 50 epochs
5. Best model saved to models/emotion_cnn_best.h5
```

### Inference Pipeline (Real-time)

```
1. Load trained model from emotion_cnn_best.h5
2. Open webcam with OpenCV
3. For each frame:
   a. Convert to grayscale
   b. Detect faces with Haar Cascade
   c. For each face:
      - Crop, resize to 48x48
      - Normalize to [0,1]
      - Pass through CNN
      - Get probability for each emotion
      - Apply temporal smoothing
      - Display result on frame
4. Show frame with predictions
5. Repeat until user presses 'q'
```

---

## Key Concepts Explained

### Why Grayscale?

FER2013 images are 48x48 grayscale. Color adds complexity without improving emotion recognition (expressions are defined by muscle movements, not skin color).

### Why BatchNormalization?

- Normalizes inputs to each layer to have mean=0, variance=1
- Allows higher learning rates (faster training)
- Reduces internal covariate shift
- Acts as regularization (slightly different normalization in train vs. test)

### Why Dropout?

During training, randomly sets a percentage of neurons to 0. This:
- Prevents co-adaptation (neurons depending on specific other neurons)
- Forces redundancy (multiple paths for each feature)
- Acts as ensemble (averaging many sub-networks)

### Why GlobalAveragePooling instead of Flatten?

- Flatten: 6x6x512 = 18,432 parameters to next layer
- GAP: 512 values (average of each 6x6 feature map)

GAP reduces parameters 36x, prevents overfitting, and is translation-invariant.

### Why Class Weights?

Dataset is imbalanced:
- Happy: 7,215 images
- Disgust: 436 images (16x fewer)

Without weights, model ignores rare classes. With weight 16.5 for disgust, a disgust mistake costs 16.5x more than a happy mistake.

---

## Files NOT Used (Legacy)

These files are from an earlier version and are NOT part of the current implementation:

| File | Original Purpose | Why Not Used |
|------|------------------|--------------|
| `emotions.ipynb` | Original training notebook | Replaced by modular train.py |
| `2nd model.ipynb` | MobileNetV2 experiment | Had bugs, replaced by model.py |
| `Dataset Prepare.ipynb` | CSV to image conversion | Dataset already prepared |
| `FERModelPrototype.h5` | Old trained model | Replaced by emotion_cnn_best.h5 |
| `fer2013.csv` | Original CSV dataset | Now using image folders |
| `happyman.jpg` | Test image | Not needed |

These files can be deleted before uploading to GitHub, or kept for historical reference.

---

## Summary

The project demonstrates a complete deep learning pipeline:

1. **Data Engineering**: Proper train/val/test splits, augmentation, class balancing
2. **Model Architecture**: Modern CNN with BatchNorm, Dropout, GAP
3. **Training Best Practices**: Callbacks for early stopping, LR scheduling, checkpointing
4. **Evaluation**: Comprehensive metrics beyond just accuracy
5. **Deployment**: Real-time inference with webcam integration

The 65.95% accuracy is competitive with published benchmarks on FER2013, and the modular code structure makes it easy to experiment with improvements.

