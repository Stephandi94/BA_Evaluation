import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TestData:
    def __init__(self):
        self.pcl_noise = np.array([
            [2, -2, -2, 1], # plane at lvl -2
            [2.5, -1, -2, 1], # aug
            [2, 0, -2, 1],
            [2, 1, -2, 1],
            # del
            [1, 2, -2, 1],
            [0, 1.5, -2, 1], # aug
            [-1, 2, -2, 1],
            [-2, 2, -2, 1],
            [-2, 1, -1.5, 1], # aug
            [-2, 0, -2, 1],
            [-2, -1, -2, 1],
            [-2, -2, -2, 1],
            [-1, -2, -2, 1],
            #del
            [1.5, -2.5, -2, 1], #aug

            [2, -2, -1, 1], # plane at lvl -1
            [2, -1, -1, 1],
            [2, 0, -0.5, 1], #aug
            [2, 1, -1, 1],
            #del
            #del
            [0, 2, -1, 1],
            [-1, 2, -1, 1],
            [-2, 4, -1, 1], #aug
            [-2, 1, -1, 1],
            [-2, 0, -1, 1],
            [-2, -1, -1, 1],
            [-2, -2, -1, 1],
            [-1.5, -2.5, -0.5, 1], # aug
            [0, -2, -1, 1],
            [1, -2, -1, 1],

            [2, -2, 0, 1], #plane at lvl 0
            [2, -1, 0, 1],
            [2, 0.5, 0, 1], # aug
            [2, 1, 0, 1],
            #del
            [1, 2, 0, 1],
            [-0.5, 2.5, 0, 1], #aug
            [-1, 2, 0, 1],
            [-2, 2, 0, 1],
            [-2, 1, 0, 1],
            [-2, 0, 0, 1],
           #del
            [-2, -2, 0, 1],
            [-1, -2, 0, 1],
            [0, -2, -1, 1], #aug
            [1, -2, 0, 1],

            [2, -2, 1, 1],  # plane at lvl 1
            [2, -1, 1, 1],
            [2, 0, 1, 1],
            [2, 1, 0.5, 1], #aug
            [2, 2, 1, 1],
            [1, 2, 1, 1],
            [0, 2, 1, 1],
            [-1, 2.5, 1, 1], #aug
            [-2, 2.5, 1, 1], #aug
            [-2, 1, 1, 1],
            #del
            #del
            [-2, -2, 1, 1],
            [-1, -2, 1, 1],
            [3, -2, 1, 1], #aug
            [1, -2, 1, 1],

            [2, -2, 2, 1],  # plane at lvl 2
            [2, -1, 2, 1],
            [3, 0, 3, 1], #aug
            [2, 1, 2, 1],
            [2, 2, 2, 1],
            [1.5, 2, 1, 1], #aug
            [0, 2, 2, 1],
            [-1, 2, 2, 1],
            [-2, 2, 2, 1],
            [-2, 1, 1, 1], #aug
            [-2, 0, 2, 1],
            [-2, -1, 2, 1],
            #del
            [-1, -2, 2, 1],
            [0, -2, 2, 1],
            [1, -2, 2, 1]
        ])

        self.pcl_no_noise = np.array([
            [2, -2, -2, 1],  # plane at lvl -2
            [2, -1, -2, 1],
            [2, 0, -2, 1],
            [2, 1, -2, 1],
            [2, 2, -2, 1],
            [1, 2, -2, 1],
            [0, 2, -2, 1],
            [-1, 2, -2, 1],
            [-2, 2, -2, 1],
            [-2, 1, -2, 1],
            [-2, 0, -2, 1],
            [-2, -1, -2, 1],
            [-2, -2, -2, 1],
            [-1, -2, -2, 1],
            [0, -2, -2, 1],
            [1, -2, -2, 1],

            [2, -2, -1, 1],  # plane at lvl -1
            [2, -1, -1, 1],
            [2, 0, -1, 1],
            [2, 1, -1, 1],
            [2, 2, -1, 1],
            [1, 2, -1, 1],
            [0, 2, -1, 1],
            [-1, 2, -1, 1],
            [-2, 2, -1, 1],
            [-2, 1, -1, 1],
            [-2, 0, -1, 1],
            [-2, -1, -1, 1],
            [-2, -2, -1, 1],
            [-1, -2, -1, 1],
            [0, -2, -1, 1],
            [1, -2, -1, 1],

            [2, -2, 0, 1],  # plane at lvl 0
            [2, -1, 0, 1],
            [2, 0, 0, 1],
            [2, 1, 0, 1],
            [2, 2, 0, 1],
            [1, 2, 0, 1],
            [0, 2, 0, 1],
            [-1, 2, 0, 1],
            [-2, 2, 0, 1],
            [-2, 1, 0, 1],
            [-2, 0, 0, 1],
            [-2, -1, 0, 1],
            [-2, -2, 0, 1],
            [-1, -2, 0, 1],
            [0, -2, 0, 1],
            [1, -2, 0, 1],

            [2, -2, 1, 1],  # plane at lvl 1
            [2, -1, 1, 1],
            [2, 0, 1, 1],
            [2, 1, 1, 1],
            [2, 2, 1, 1],
            [1, 2, 1, 1],
            [0, 2, 1, 1],
            [-1, 2, 1, 1],
            [-2, 2, 1, 1],
            [-2, 1, 1, 1],
            [-2, 0, 1, 1],
            [-2, -1, 1, 1],
            [-2, -2, 1, 1],
            [-1, -2, 1, 1],
            [0, -2, 1, 1],
            [1, -2, 1, 1],

            [2, -2, 2, 1],  # plane at lvl 2
            [2, -1, 2, 1],
            [2, 0, 2, 1],
            [2, 1, 2, 1],
            [2, 2, 2, 1],
            [1, 2, 2, 1],
            [0, 2, 2, 1],
            [-1, 2, 2, 1],
            [-2, 2, 2, 1],
            [-2, 1, 2, 1],
            [-2, 0, 2, 1],
            [-2, -1, 2, 1],
            [-2, -2, 2, 1],
            [-1, -2, 2, 1],
            [0, -2, 2, 1],
            [1, -2, 2, 1]
        ])

    def get_gt(self):
        return self.pcl_no_noise

    def get_noisy(self):
        return self.pcl_noise

    def plot_gt(self):
        fig = plt.figure()
        ax = Axes3D(fig)#plt.axes(projection='3d')

        x = self.pcl_no_noise[:, 0]
        y = self.pcl_no_noise[:, 1]
        z = self.pcl_no_noise[:, 2]
        ax.scatter(x, y, z, c='r')
        plt.show()

    def plot_noise(self):
        fig = plt.figure()
        ax = Axes3D(fig)#plt.axes(projection='3d')

        x = self.pcl_noise[:, 0]
        y = self.pcl_noise[:, 1]
        z = self.pcl_noise[:, 2]
        ax.scatter(x, y, z, c='b')
        plt.show()

    def plot_layover(self, pcl1, pcl2):
        fig = plt.figure()
        ax = Axes3D(fig)#plt.axes(projection='3d')

        x = pcl1[:, 0]
        y = pcl1[:, 1]
        z = pcl1[:, 2]
        ax.scatter(x, y, z, c='r')

        x = pcl2[:, 0]
        y = pcl2[:, 1]
        z = pcl2[:, 2]
        ax.scatter(x, y, z, c='b')

        plt.show()