# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv1D, Flatten, Dense
from scipy.stats import linregress, t
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers

from Utils.utils import normalize_dataset

np.random.seed(9)

# Define parameters
train_ratio = 0.8
variety = 'burbank'
assert variety in ['burbank', 'mountain_gem']

# Load dataset and normalize data
dataset = np.loadtxt(f'PDT detection/Dataset Tuberosum/Dimensions/{variety}.txt', skiprows=2)
data, labels = normalize_dataset(dataset.T[:2].T), dataset.T[2]

# Shuffle data and target
perm = np.random.permutation(len(data))
data = data[perm]
labels = labels[perm]

# Split train/test samples
n = int(train_ratio * len(data))
train, train_labels = data[:n], labels[:n]
test, test_labels = data[n:], labels[n:]

# Define the K-fold Cross Validator
kfold = KFold(n_splits=16, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
rmse, mbe, nse, mae, totW, totWp = [], [], [], [], [], []
for split_train, split_test in kfold.split(data, labels):
    # Define model
    model = keras.Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')

    callbacks = [keras.callbacks.ModelCheckpoint(f"Trained Models/{variety}_weight.h5", save_best_only=True),
                 keras.callbacks.EarlyStopping(patience=20)]

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    model.fit(train, train_labels, epochs=1000, validation_data=(test, test_labels), batch_size=4, verbose=0,
              shuffle=True, callbacks=callbacks)
    model.save(f'Trained Models/{variety}_weight.h5')

    # Generate generalization metrics
    preds = [i[0] for i in model.predict(data[split_test])]
    loss = model.evaluate(data[split_test], labels[split_test], verbose=0)
    bias = labels[split_test] - preds
    mean_abs = np.mean([abs(i) for i in bias])
    nash_num = np.sum([(i - j) ** 2 for i, j in zip(labels[split_test], preds)])
    nash_den = np.sum([(i - np.mean(labels[split_test])) ** 2 for i in labels[split_test]])
    nash = 1 - nash_num / nash_den
    print(
        f'Score for fold {fold_no}: RMSE of {np.sqrt(loss): .2f} g | MAE of {mean_abs:.2f} g | MBE {np.mean(bias):.2f} | NSE of {nash:.2f}')
    rmse.append(np.sqrt(loss))
    mbe.append(np.mean(bias))
    nse.append(nash)
    mae.append(mean_abs)
    totW.append(np.abs(np.sum(test_labels) - np.sum([i[0] for i in model.predict(test)])))
    totWp.append(np.abs(np.sum(test_labels) - np.sum([i[0] for i in model.predict(test)]))/np.sum(test_labels))
    # Increase fold number
    fold_no = fold_no + 1

tinv = lambda p, df: abs(t.ppf(p / 2, df))
print(f"RMSE (95%): ({np.mean(rmse):.2f}+/-{tinv(0.05, len(rmse) - 2) * np.std(rmse) / np.sqrt(len(rmse)):.2f}) g")
print(f"MAE (95%): ({np.mean(mae):.2f}+/-{tinv(0.05, len(mae) - 2) * np.std(mae) / np.sqrt(len(mae)):.2f}) g")
print(f"MBE  (95%): ({np.mean(mbe):.2f}+/-{tinv(0.05, len(mbe) - 2) * np.std(mbe) / np.sqrt(len(mbe)):.2f}) g")
print(f"NSE  (95%): ({np.mean(nse):.2f}+/-{tinv(0.05, len(nse) - 2) * np.std(nse) / np.sqrt(len(nse)):.2f})")
print(f"Poids total écart abs. (95%): ({np.mean(totW):.2f}+/-{tinv(0.05, len(totW) - 2) * np.std(totW) / np.sqrt(len(totW)):.2f})")
print(f"Poids total écart rel. (95%): ({np.mean(totWp)*100:.2f}+/-{tinv(0.05, len(totWp) - 2) * np.std(totWp) / np.sqrt(len(totWp)):.2f})")

# Load the model and evaluate
model = keras.models.load_model(f"Trained Models/{variety}_weight.h5")
preds_train = model.predict(train)
preds_test = model.predict(test)

# Linear regression
res_train = linregress(train_labels, [i[0] for i in preds_train])
res_test = linregress(test_labels, [i[0] for i in preds_test])

# Display results
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

# Scatter plots for all (true, pred) points
ax1.scatter(train_labels, [i[0] for i in preds_train])
ax2.scatter(test_labels, [i[0] for i in preds_test], c='red')
ax1.set_xlim(0, 600)
ax1.set_ylim(0, 600)
ax2.set_xlim(0, 600)
ax2.set_ylim(0, 600)

# Plot linear regression results
ax1.plot(train_labels, [res_train.slope * i + res_train.intercept for i in train_labels],
         label='Régression linéaire y=ax+b\n'
               f'a={res_train.slope:.2f}$\pm${res_train.stderr:.2f}\n'
               f'b={res_train.intercept:.2f}$\pm${res_test.intercept_stderr:.2f}\n'
               f'R$^2$={res_train.rvalue ** 2:.3f}')
ax2.plot(test_labels, [res_test.slope * i + res_test.intercept for i in test_labels], 'red',
         label='Régression linéaire y=ax+b\n'
               f'a={res_test.slope:.2f}$\pm${res_test.stderr:.2f}\n'
               f'b={res_test.intercept:.2f}$\pm${res_test.intercept_stderr:.2f}\n'
               f'R$^2$={res_test.rvalue ** 2:.3f}')

ax1.set_title("Données d'entraînement")
ax1.set_xlabel('Poids réel [g]')
ax1.set_ylabel('Poids prédit [g]')
ax1.legend()

ax2.set_title("Données de test")
ax2.set_xlabel('Poids réel [g]')
ax2.set_ylabel('Poids prédit [g]')
ax2.legend()

plt.tight_layout()
# plt.savefig(f'{variety}.png', dpi=500)
plt.show()
