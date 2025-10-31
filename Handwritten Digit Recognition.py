import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_and_preprocess_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize to range [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

  
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return (x_train, y_train_cat), (x_test, y_test_cat), (y_train, y_test)

def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> models.Model:
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def compile_and_train_model(model: models.Model, 
                            x_train: np.ndarray, 
                            y_train: np.ndarray, 
                            x_test: np.ndarray, 
                            y_test: np.ndarray) -> models.Model:
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=64,
              validation_data=(x_test, y_test))
    return model

def evaluate_model(model: models.Model, x_test: np.ndarray, y_test: np.ndarray) -> None:
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"📉 Test Loss: {test_loss:.4f}")

def display_predictions(model: models.Model, x_test: np.ndarray, y_test: np.ndarray, y_true_labels: np.ndarray, num_images: int = 10):
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, 5, i+1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_true_labels[i]}\nPred: {predicted_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    return predicted_labels

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix for MNIST CNN Model")
    plt.show()

if __name__ == "__main__":
    (x_train, y_train_cat), (x_test, y_test_cat), (y_train, y_test) = load_and_preprocess_data()
    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=10)
    model = compile_and_train_model(model, x_train, y_train_cat, x_test, y_test_cat)
    evaluate_model(model, x_test, y_test_cat)

    # Display predictions and confusion matrix
    y_pred = display_predictions(model, x_test, y_test_cat, y_test, num_images=10)
    plot_confusion_matrix(y_test, y_pred)
