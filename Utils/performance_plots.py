import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14})

epochs15 = np.linspace(1, 15, 15, dtype=np.uint8)
epochs20 = np.linspace(1, 20, 20, dtype=np.uint8)

# metrics from binary mask training
dice1 = [0.5591, 0.9538, 0.9693, 0.9745, 0.9762, 0.9773, 0.9784, 0.9785, 0.9796, 0.9802, 0.9812, 0.9815, 0.9825, 0.9834, 0.9840]
val_dice1 = [0.8919, 0.9672, 0.9728, 0.9750, 0.9756, 0.9764, 0.9726, 0.9779, 0.9782, 0.9796, 0.9807, 0.9795, 0.9814, 0.9818, 0.9820]
acc1 = [0.4055, 0.9645, 0.9765, 0.9805, 0.9818, 0.9826, 0.9834, 0.9835, 0.9844, 0.9849, 0.9856, 0.9859, 0.9866, 0.9873, 0.9877]
val_acc1 = [0.9088, 0.9750, 0.9792, 0.9805, 0.9810, 0.9815, 0.9792, 0.9827, 0.9831, 0.9842, 0.9850, 0.9842, 0.9856, 0.9859, 0.9861]
loss1 = [1-i for i in dice1]
val_loss1 = [1-i for i in val_dice1]

# metrics from contours training with only dice
dice2 = [0.5616, 0.7389, 0.7674, 0.7856, 0.7998, 0.8118, 0.8234, 0.8339, 0.8434, 0.8489, 0.8543, 0.8619, 0.8675, 0.8728, 0.8758, 0.8807, 0.8817, 0.8861, 0.8900, 0.8920]
val_dice2 = [0.7212, 0.7586, 0.7745, 0.7864, 0.7961, 0.8057, 0.8073, 0.8185, 0.8233, 0.8298, 0.8268, 0.8414, 0.8465, 0.8485, 0.8495, 0.8553, 0.8565, 0.8593, 0.8587, 0.8642]
acc2 = [0.8866, 0.9617, 0.9662, 0.9691, 0.9712, 0.9731, 0.9748, 0.9764, 0.9779, 0.9786, 0.9795, 0.9806, 0.9814, 0.9822, 0.9826, 0.9833, 0.9835, 0.9841, 0.9847, 0.9849]
val_acc2 = [0.9565, 0.9644, 0.9672, 0.9686, 0.9710, 0.9723, 0.9727, 0.9735, 0.9742, 0.9751, 0.9754, 0.9772, 0.9779, 0.9784, 0.9791, 0.9797, 0.9803, 0.9808, 0.9806, 0.9809]
loss2 = [1-i for i in dice2]
val_loss2 = [1-i for i in val_dice2]

# metrics from contours with dice and weighted BCE
dice3 = [0.3414, 0.4880, 0.5459, 0.5849, 0.6094, 0.6380, 0.6526, 0.6218, 0.6309, 0.6859, 0.7052, 0.7197, 0.7189, 0.7235, 0.7398, 0.7497, 0.7575, 0.7618, 0.7698, 0.7702]
val_dice3 = [0.4661, 0.5374, 0.5890, 0.5995, 0.6155, 0.6487, 0.6436, 0.4831, 0.6700, 0.6870, 0.7013, 0.7261, 0.7086, 0.7335, 0.7583, 0.7567, 0.7624, 0.7418, 0.7462, 0.7491]
acc3 = [0.6903, 0.8701, 0.8930, 0.9069, 0.9148, 0.9241, 0.9285, 0.9096, 0.9218, 0.9385, 0.9437, 0.9475, 0.9472, 0.9484, 0.9525, 0.9549, 0.9567, 0.9577, 0.9596, 0.9597]
val_acc3 = [0.8559, 0.8909, 0.9098, 0.9111, 0.9160, 0.9275, 0.9274, 0.8725, 0.9340, 0.9389, 0.9427, 0.9494, 0.9448, 0.9514, 0.9577, 0.9572, 0.9587, 0.9531, 0.9541, 0.9548]
loss3 = [1.8663, 1.1986, 1.0118, 0.8942, 0.8264, 0.7527, 0.7180, 0.8291, 0.7750, 0.6373, 0.5931, 0.5601, 0.5601, 0.5645, 0.5536, 0.5168, 0.4948, 0.4797, 0.4522, 0.4527]
val_loss3 = [1.2631, 1.0559, 0.9322, 0.8820, 0.8423, 0.8355, 0.9732, 1.1772, 0.7465, 0.7719, 0.7101, 0.7141, 0.7137, 0.7094, 0.7055, 0.7113, 0.7370, 0.6590, 0.6651, 0.6359]

fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(epochs15, loss1, 'r', label='Entraînement masque')
ax.plot(epochs15, val_loss1, 'r--', label='Évaluation masque')
ax.plot(epochs20, loss2, 'm', label='Entraînement contours Dice')
ax.plot(epochs20, val_loss2, 'm--', label='Évaluation contours Dice')
ax.plot(epochs20, loss3, 'g', label='Entraînement contours Dice+BCE')
ax.plot(epochs20, val_loss3, 'g--', label='Évaluation contours Dice+BCE')

ax.set_yscale('log')
# ax.set_yticks([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0])
# ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylim(0.004, 2)

ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylabel('Loss', fontsize=16)
ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])


ax.legend(loc='lower right', fontsize=12)

plt.tight_layout()
fig.savefig('losses.png', dpi=600)
plt.show()

fig, ax = plt.subplots(ncols=2, figsize=(14, 6))
ax = ax.ravel()

# ax.plot(epochs, loss, 'b', label='Entraînement')
# ax.plot(epochs, val_loss, 'b--', label='Évaluation')

ax[0].plot(epochs15, dice1, 'r', label='Entraînement masque')
ax[0].plot(epochs15, val_dice1, 'r--', label='Évaluation masque')
ax[0].plot(epochs20, dice2, 'm', label='Entraînement contours DSC')
ax[0].plot(epochs20, val_dice2, 'm--', label='Évaluation contours DSC')
ax[0].plot(epochs20, dice3, 'g', label='Entraînement contours DSC+BCE')
ax[0].plot(epochs20, val_dice3, 'g--', label='Évaluation contours DSC+BCE')
ax[0].set_xlabel('Epoch', fontsize=16)
ax[0].set_ylabel('Coefficient Dice', fontsize=16)

ax[1].plot(epochs15, acc1, 'r', label='Entraînement masque')
ax[1].plot(epochs15, val_acc1, 'r--', label='Évaluation masque')
ax[1].plot(epochs20, acc2, 'm', label='Entraînement contours DSC')
ax[1].plot(epochs20, val_acc2, 'm--', label='Évaluation contours DSC')
ax[1].plot(epochs20, acc3, 'g', label='Entraînement contours DSC+BCE')
ax[1].plot(epochs20, val_acc3, 'g--', label='Évaluation contours DSC+BCE')
ax[1].set_xlabel('Epoch', fontsize=16)
ax[1].set_ylabel('Exactitude', fontsize=16)

# ax.set_yscale('log')
ax[0].set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
ax[1].set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
# ax.set_yticks([0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
# ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].set_ylim(0, 1)
ax[1].set_ylim(0, 1)

ax[0].legend(loc='lower right', fontsize=12)
ax[1].legend(loc='lower right', fontsize=12)

plt.tight_layout()
plt.show()
fig.savefig('comparison.png', dpi=600)