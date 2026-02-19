import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(128,128,3)),
    layers.Flatten(),
    layers.Dense(3, activation="softmax")
])

model.save("rice_model.keras")
print("Model saved!")


