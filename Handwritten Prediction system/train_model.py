"""
Handwritten Digit Recognition - Model Training Script
Train a deep learning model (Multi-layer Perceptron) on MNIST dataset
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("HANDWRITTEN DIGIT RECOGNITION - MODEL TRAINING")
print("=" * 70)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load training data
print("\n[1/8] Loading training data...")
train_df = pd.read_csv('train.csv')
print(f"   ✓ Training data loaded: {train_df.shape}")

# Separate features and labels
X = train_df.drop('label', axis=1).values
y = train_df['label'].values

print(f"   ✓ Features shape: {X.shape}")
print(f"   ✓ Labels shape: {y.shape}")
print(f"   ✓ Number of classes: {len(np.unique(y))}")

# Normalize pixel values (0-255 to 0-1)
print("\n[2/8] Normalizing data...")
X = X / 255.0
print("   ✓ Data normalized to range [0, 1]")

# Split data into train and validation sets
print("\n[3/8] Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Training set: {X_train.shape[0]} samples")
print(f"   ✓ Validation set: {X_val.shape[0]} samples")

# Build the Multi-layer Perceptron model
print("\n[4/8] Building Deep Learning Model...")
model = keras.Sequential([
    # Input layer
    layers.Input(shape=(784,)),
    
    # First hidden layer
    layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Second hidden layer
    layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Third hidden layer
    layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # Fourth hidden layer
    layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # Output layer
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("   ✓ Model architecture:")
model.summary()

# Define callbacks
print("\n[5/8] Setting up training callbacks...")
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]
print("   ✓ Callbacks configured: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau")

# Train the model
print("\n[6/8] Training model...")
print("   This may take several minutes...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# Load the best model
model = keras.models.load_model('best_model.h5')
print("\n   ✓ Best model loaded")

# Evaluate on validation set
print("\n[7/8] Evaluating model...")
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"   ✓ Validation Loss: {val_loss:.4f}")
print(f"   ✓ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# Make predictions
y_pred = model.predict(X_val, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate metrics
print("\n   Detailed Classification Report:")
print(classification_report(y_val, y_pred_classes, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred_classes)
print("\n   Confusion Matrix:")
print(cm)

# Save confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - Handwritten Digit Recognition', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✓ Confusion matrix saved: confusion_matrix.png")

# Save training history plot
plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("   ✓ Training history saved: training_history.png")

# Save model metrics
metrics = {
    'accuracy': float(val_accuracy),
    'loss': float(val_loss),
    'classification_report': classification_report(y_val, y_pred_classes, output_dict=True),
    'confusion_matrix': cm.tolist(),
    'training_samples': int(X_train.shape[0]),
    'validation_samples': int(X_val.shape[0]),
    'epochs_trained': len(history.history['loss'])
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("   ✓ Model metrics saved: model_metrics.json")

# Save the final model
print("\n[8/8] Saving model...")
model.save('digit_recognition_model.h5')
print("   ✓ Model saved: digit_recognition_model.h5")

# Save model in TensorFlow SavedModel format (for better compatibility)
model.save('digit_recognition_model', save_format='tf')
print("   ✓ Model saved: digit_recognition_model/ (TensorFlow format)")

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\n📊 Final Results:")
print(f"   • Accuracy: {val_accuracy*100:.2f}%")
print(f"   • Loss: {val_loss:.4f}")
print(f"   • Epochs Trained: {len(history.history['loss'])}")
print(f"\n📁 Generated Files:")
print(f"   • digit_recognition_model.h5")
print(f"   • digit_recognition_model/ (folder)")
print(f"   • model_metrics.json")
print(f"   • confusion_matrix.png")
print(f"   • training_history.png")
print("\n✅ Ready to use with Streamlit app!")
print("=" * 70)
