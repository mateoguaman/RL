import numpy as np
import matplotlib.pyplot as plt

depthS = np.zeros((10, 1000))
depthSPoly = np.zeros((10, 1000))
depthQ = np.zeros((10, 1000))
depthQPoly = np.zeros((10, 1000))


for i in range(10):
    filename = "/Users/mateoguaman/eclipse-workspace/newPacman2/depthS" + str(i + 1) + ".csv"
    depthS[i, :] = np.genfromtxt(filename, delimiter=',')[:, 1]
mean_depthS = np.mean(depthS, axis=0)
std_depthS = np.std(depthS, axis=0)

for i in range(10):
    filename = "/Users/mateoguaman/eclipse-workspace/newPacman2/depthS" + str(i + 1) + "Poly.csv"
    depthSPoly[i, :] = np.genfromtxt(filename, delimiter=',')[:, 1]
mean_depthSPoly = np.mean(depthSPoly, axis=0)
std_depthSPoly = np.std(depthSPoly, axis=0)

for i in range(10):
    filename = "/Users/mateoguaman/eclipse-workspace/newPacman2/depthQ" + str(i + 1) + ".csv"
    depthQ[i, :] = np.genfromtxt(filename, delimiter=',')[:, 1]
mean_depthQ = np.mean(depthQ, axis=0)
std_depthQ = np.std(depthQ, axis=0)

for i in range(10):
    filename = "/Users/mateoguaman/eclipse-workspace/newPacman2/depthQ" + str(i + 1) + "Poly.csv"
    depthQPoly[i, :] = np.genfromtxt(filename, delimiter=',')[:, 1]
mean_depthQPoly = np.mean(depthQPoly, axis=0)
std_depthQPoly = np.std(depthQPoly, axis=0)


depthS_plot, = plt.plot(np.arange(1000), mean_depthS, '-', color='gray', label='depthS plot')
plt.fill_between(np.arange(1000), mean_depthS - std_depthS, mean_depthS + std_depthS, color='gray', alpha=0.3)
plt.title("Score vs Episodes, depthS")
plt.ylabel("Score (Expected reward)")
plt.xlabel("Episode")
plt.legend(handles=[depthS_plot])
plt.ylim(top=4000)
plt.savefig('depthS.png', dpi=300, bbox_inches='tight')
plt.close()

depthSPoly_plot, = plt.plot(np.arange(1000), mean_depthSPoly, '-', color='blue', label='depthSPoly plot')
plt.fill_between(np.arange(1000), mean_depthSPoly - std_depthSPoly, mean_depthSPoly + std_depthSPoly, color='blue', alpha=0.3)
plt.title("Score vs Episodes, depthS with Polynomial features")
plt.ylabel("Score (Expected reward)")
plt.xlabel("Episode")
plt.legend(handles=[depthSPoly_plot])
plt.ylim(top=4000)
plt.savefig('depthSPoly.png', dpi=300, bbox_inches='tight')
plt.close()

depthQ_plot, = plt.plot(np.arange(1000), mean_depthQ, '-', color='red', label="depthQ plot")
plt.fill_between(np.arange(1000), mean_depthQ - std_depthQ, mean_depthQ + std_depthQ, color='red', alpha=0.3)
plt.title("Score vs Episodes, depthQ")
plt.ylabel("Score (Expected reward)")
plt.xlabel("Episode")
plt.legend(handles=[depthQ_plot])
plt.ylim(top=4000)
plt.savefig('depthQ.png', dpi=300, bbox_inches='tight')
plt.close()

depthQPoly_plot, = plt.plot(np.arange(1000), mean_depthQPoly, '-', color='brown', label='depthSPoly plot')
plt.fill_between(np.arange(1000), mean_depthQPoly - std_depthQPoly, mean_depthQPoly + std_depthQPoly, color='brown', alpha=0.3)
plt.title("Score vs Episodes, depthQ with Polynomial features")
plt.ylabel("Score (Expected reward)")
plt.xlabel("Episode")
plt.legend(handles=[depthQPoly_plot])
plt.ylim(top=4000)
plt.savefig('depthQPoly.png', dpi=300, bbox_inches='tight')
plt.close()
