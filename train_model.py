from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===============================
# Image Data Generator (Augmentation)
# ===============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True
)

# ===============================
# Training Generator
# ===============================
train_gen = datagen.flow_from_directory(
    'dataset/',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("Class Indices:", train_gen.class_indices)

# ===============================
# Validation Generator
# ===============================
val_gen = datagen.flow_from_directory(
    'dataset/',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False   # 🔥 IMPORTANT
)

# ===============================
# CNN Model
# ===============================
model = Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(len(train_gen.class_indices), activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===============================
# Callbacks
# ===============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor='val_accuracy',
    save_best_only=True
)

# ===============================
# Train Model
# ===============================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[early_stop, checkpoint]
)

# ===============================
# Save Model
# ===============================
model.save("sign_model.h5")
print("Model trained and saved as sign_model.h5")

# ===============================
# ROC Curve
# ===============================
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# Reset generator
val_gen.reset()

# Predictions
y_pred_prob = model.predict(val_gen)

# True labels
y_true = val_gen.classes

# Number of classes
n_classes = len(train_gen.class_indices)

# Convert to binary
y_true_bin = label_binarize(y_true, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

# Calculate ROC
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()