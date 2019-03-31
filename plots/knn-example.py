import matplotlib.pyplot as plt
import numpy as np



x1 = [1, 1, 2, 4, 1.5, 2.5, 3, 4]
y1 = [3, 4, 1, 4, 2.5, 3.5, 1.5, 2]

x2 = [9.5, 8, 6, 6, 7.5, 8, 8.5, 7]
y2 = [7, 9, 7, 6, 7.5, 9, 9.5, 8.5]



fig = plt.figure()

plt.xlim(0,10)
plt.ylim(0,10)

plt.plot([5.5,6],[5.5,6], c='grey')
plt.plot([5.5,6],[5.5,7], c='grey')
plt.plot([5.5,4],[5.5,4], c='grey')

plt.scatter(x1,y1, c="red")
plt.scatter(x2,y2, c="blue")
plt.scatter([5.5],[5.5], c='k', marker="x")

# plt.title("K-nearest Neighbors 2D example")

# plt.show()
plt.savefig("./images/knn_example.svg")
