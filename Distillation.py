import os
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from Utils.utils import normalize_dataset


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=0.1,
            temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


np.random.seed(9)

# Define parameters
train_ratio = 0.8
variety = 'burbank'
assert variety in ['burbank', 'mountain_gem']

# Load dataset and normalize data
dataset = np.loadtxt(f'PDT detection/SolanumTuberosum/Dimensions/{variety}.txt', skiprows=2)
data, labels = normalize_dataset(dataset.T[:2].T), dataset.T[2]

# Shuffle data and target
perm = np.random.permutation(len(data))
data = data[perm]
labels = labels[perm]

# Split train/test samples
n = int(train_ratio * len(data))
train, train_labels = data[:n], labels[:n]
test, test_labels = data[n:], labels[n:]

# Define the models
teacher = keras.Sequential([
        layers.Dense(128, activation='relu'),
        layers.GaussianNoise(0.01),
        layers.Dense(128, activation='relu'),
        layers.GaussianNoise(0.01),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ], name='teacher')
student = keras.Sequential([
        layers.Dense(32, activation='relu'),
        layers.GaussianNoise(0.01),
        layers.Dense(32, activation='relu'),
        layers.GaussianNoise(0.01),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ], name='student')
student_scratch = keras.models.clone_model(student)

# Train the teacher
callbacks = [keras.callbacks.ModelCheckpoint(f"Trained Models/distiller.h5", save_best_only=True), keras.callbacks.EarlyStopping(patience=20)]
teacher.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
print('TRAINING TEACHER...')
teacher.fit(train, train_labels, epochs=400, validation_data=(test, test_labels), batch_size=4, shuffle=True, callbacks=callbacks)

# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=['mean_squared_error', 'mean_absolute_error'],
    student_loss_fn='mean_squared_error',
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# Distill teacher to student
print('TRAINING DISTILLER...')
distiller.fit(train, train_labels, epochs=400, validation_data=(test, test_labels), batch_size=4, shuffle=True, callbacks=callbacks)

# Scratch student
student_scratch.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
print('TRAINING STUDENT FROM SCRATCH...')
student_scratch.fit(train, train_labels, epochs=400, validation_data=(test, test_labels), batch_size=4, shuffle=True, callbacks=callbacks)