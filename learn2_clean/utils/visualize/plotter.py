import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# matplotlib.use('TkAgg')


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


class Plotting:
    def __init__(self):
        self.data = np.empty((0, 3))
        self.model = np.empty((0, 3))
        self.scores = [1, 4, 9, 16]
        self.time = [1, 2, 3, 4]
        self.iterations = [1, 2, 3, 4]
        self.modeler_name = 'modeler'
        self.current_iteration = 0
        self.env_id = 0
        self.best_score = 0.
        self.model_color = None
        self.model_image = None


class Plotter:

    def __init__(self, compared_file):
        if compared_file is None:
            self.compared = None
            return
        try:
            with open(compared_file, 'rb') as f:
                self.compared = pickle.load(f)
        except (EOFError, FileNotFoundError) as e:
            self.compared = None
            print(e)

    def update(self, pipe):
        if pipe.poll():
            plotting = pipe.recv()
            self.plot(plotting)

    def __call__(self, pipe=None):
        # matplotlib.use('TkAgg')
        print("backend is " + matplotlib.get_backend())

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig2 = plt.figure()
        # self.ax2 = self.fig2.add_subplot(222)
        # self.fig3 = plt.figure()
        self.ax3 = self.fig2.add_subplot(121)
        self.ax4 = self.fig2.add_subplot(122)

        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111, projection='3d')
        # # self.fig2 = plt.figure()
        # # self.ax2 = self.fig2.add_subplot(111)
        # self.fig3 = plt.figure()
        # self.ax3 = self.fig3.add_subplot(111)
        # self.fig4 = plt.figure()
        # self.ax4 = self.fig4.add_subplot(111)

        if pipe is None:
            plt.ion()
        else:
            timer = self.fig.canvas.new_timer(interval=1000)
            timer.add_callback(self.update, pipe)
            timer.start()
        plt.show()

    def plot(self, plotting=Plotting()):
        self.ax.cla()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        data = plotting.data
        model = plotting.model
        if data.size > 0:
            if data.shape[1] == 3:
                self.ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='.', s=10.)
            elif data.shape[1] == 2:
                z = np.zeros(data.shape[0])
                self.ax.scatter(data[:, 0], data[:, 1], z, c='r', marker='.', s=10.)
            else:
                assert False
        if model.size > 0:
            if model.shape[1] == 3:
                if plotting.model_color is None:
                    self.ax.scatter(model[:, 0], model[:, 1], model[:, 2], c='g', marker='.', s=10.)
                else:
                    assert plotting.model_color.shape[0] == model.shape[0]
                    self.ax.scatter(model[:, 0], model[:, 1], model[:, 2], c=plotting.model_color, marker='.', s=100.)
            elif model.shape[1] == 2:
                z = np.zeros(model.shape[0])
                if plotting.model_color is None:
                    self.ax.scatter(model[:, 0], model[:, 1], z, c='g', marker='.', s=10.)
                else:
                    assert plotting.model_color.shape[0] == model.shape[0]
                    self.ax.scatter(model[:, 0], model[:, 1], z, c=plotting.model_color, marker='.', s=100.)
            else:
                assert False
        set_axes_equal(self.ax)
        self.fig.suptitle(f'env_id: {plotting.env_id}, iteration: {plotting.current_iteration}, '
                          f'score: {plotting.best_score:.4f}')
        self.fig.canvas.draw()

        # if plotting.model_image is not None:
        #     self.ax2.cla()
        #     self.ax2.imshow(plotting.model_image)

        self.ax3.cla()
        self.ax3.set_xlabel("time (s)")
        self.ax3.set_ylabel("score")
        self.ax3.plot(plotting.time, plotting.scores, 'r', label=plotting.modeler_name)
        if self.compared is not None:
            self.ax3.plot(self.compared['evolved_time'], self.compared['evolved_scores'], 'g',
                          label=self.compared['cfg'].modeler_name)
        self.ax3.legend() # 添加图例
        self.fig2.canvas.draw()

        self.ax4.cla()
        self.ax4.set_xlabel("iteration")
        self.ax4.set_ylabel("score")
        self.ax4.plot(plotting.iterations, plotting.scores, 'r', label=plotting.modeler_name)
        if self.compared is not None:
            self.ax4.plot(self.compared['evolved_iterations'], self.compared['evolved_scores'], 'g',
                          label=self.compared['cfg'].modeler_name)
        self.ax4.legend() # 添加图例
        self.fig2.canvas.draw()

        # self.fig2.canvas.draw()


