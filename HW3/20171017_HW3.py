import numpy as np
from matplotlib import pyplot as plt

x1 = np.linspace(0, 1, 1000)
x2 = np.random.uniform(low=-0.01, high=0.01, size=(1000,)) + x1

data = np.array([x1, x2]).reshape((2, 1000))

eigen_values, eigen_vectors = np.linalg.eig(np.cov(data))

mean = data.T.mean(axis=0)

standard_deviation = np.dot(data.T, eigen_vectors).std(axis=0).mean()

figure, axis = plt.subplots(figsize=(25, 25))

plt.scatter(data[0, :], data[1, :], color='pink')

for i in range(2):
    eigen_vector = eigen_vectors[i]
    start = mean
    end = mean - standard_deviation * eigen_vector

    if i == 0:
        axis.annotate(
            '',
            xy=end,
            xycoords='data',
            xytext=start,
            textcoords='data',
            arrowprops=dict(facecolor='red', width=2.0)
        )

    elif i == 1:
        axis.annotate(
            '',
            xy=end,
            xycoords='data',
            xytext=start,
            textcoords='data',
            arrowprops=dict(facecolor='green', width=2.0)
        )

axis.set_aspect('equal')
plt.show()
