import matplotlib.pyplot as plt
import numpy as np


N = 20

x = np.random.rand(N)
y = np.random.rand(N)

colors = np.random.rand(N)

area = np.pi * (45 * np.random.rand(N))**2


plt.scatter(x, y, s=area, c=colors, alpha=0.4, label="Test label")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
