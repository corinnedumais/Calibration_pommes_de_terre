import matplotlib.pyplot as plt
import numpy as np

gem_file = np.loadtxt('SolanumTuberosum/Dimensions/mountain_gem.txt', skiprows=2)
bank_file = np.loadtxt('SolanumTuberosum/Dimensions/burbank.txt', skiprows=2)

length_gem, width_gem = gem_file.T[0], gem_file.T[1]
length_bank, width_bank = bank_file.T[0], bank_file.T[1]

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey='all', figsize=(10, 4))
ax1.hist(length_bank, bins=[50, 70, 90, 110, 130, 150, 170, 190], alpha=0.5, histtype='bar', ec='black')
ax2.hist(width_bank, bins=[20, 30, 40, 50, 60, 70, 80, 90, 100], alpha=0.5, histtype='bar', ec='black')
ax1.set_xlabel('Longueur réelle [mm]')
ax2.set_xlabel('Largeur réelle [mm]')
plt.show()