import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import linregress

from tensorflow.keras import utils, layers
from tensorflow import keras

from Utils.weights import normalize_dataset

np.random.seed(6)

# Define parameters
train_ratio = 0.7
variety = 'mountain_gem'  # must be 'burbank' of 'mountain_gem'

# Load dataset and normalize data
dataset = np.loadtxt(f'SolanumTuberosum/Dimensions/{variety}.txt', skiprows=2)
data, labels = normalize_dataset(dataset.T[:2].T), dataset.T[2]

# Shuffle data and target
perm = np.random.permutation(len(data))
data = data[perm]
labels = labels[perm]

# Split train/test samples
n = int(train_ratio * len(data))
train, train_labels = data[:n], labels[:n]
test, test_labels = data[n:], labels[n:]

# Define model
model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.GaussianNoise(0.01),
    layers.Dense(128, activation='relu'),
    layers.GaussianNoise(0.01),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

callbacks = [keras.callbacks.ModelCheckpoint(f"Trained Models/{variety}_weight.h5", save_best_only=True)]

# Train the model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
model.fit(train, train_labels, epochs=300, validation_data=(test, test_labels), batch_size=4, shuffle=True, callbacks=callbacks)

# Load the model and evaluate
model = keras.models.load_model(f"Trained Models/{variety}_weight.h5")
preds_train = model.predict(train)
preds_test = model.predict(test)

# Print expected values vs predictions
print('\nTRAIN EVALUATION')
for label, pred in zip(train_labels, preds_train):
    print(f'Expected value: {label:.1f} g      Predicted value: {pred[0]:.1f} g')

print('\nTEST EVALUATION')
for label, pred in zip(test_labels, preds_test):
    print(f'Expected value: {label:.1f} g      Predicted value: {pred[0]:.1f} g')

# Linear regression
res_train = linregress(train_labels, [i[0] for i in preds_train])
res_test = linregress(test_labels, [i[0] for i in preds_test])

# Display results
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
ax1.scatter(train_labels, [i[0] for i in preds_train])
ax1.plot(train_labels, [res_train.slope * i + res_train.intercept for i in train_labels], label='Régression linéaire\n'
                                                                                                f'y={res_train.slope:.2f}x+{res_train.intercept:.2f}\n'
                                                                                                f'R$^2$={res_train.rvalue ** 2:.4f}')
ax2.scatter(test_labels, [i[0] for i in preds_test], c='red')
ax2.plot(test_labels, [res_test.slope * i + res_test.intercept for i in test_labels], 'red', label='Régression linéaire\n'
                                                                                                f'y={res_test.slope:.2f}x+{res_test.intercept:.2f}\n'
                                                                                                f'R$^2$={res_test.rvalue ** 2:.4f}')
ax1.set_title("Données d'entraînement")
ax1.set_xlabel('Poids réel [g]')
ax1.set_ylabel('Poids prédit [g]')
ax1.legend()

ax2.set_title("Données de test")
ax2.set_xlabel('Poids réel [g]')
ax2.set_ylabel('Poids prédit [g]')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{variety}.png', dpi=500)
plt.show()