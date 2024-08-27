import matplotlib.pyplot as plt
import numpy as np


def compare_point_clouds(cloud1,  cloud2, color1='b', color2='g'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter(cloud1[:, 0], cloud1[:, 1], cloud1[:, 2], s=1, c=color1, marker='o')
    ax.scatter(cloud2[:, 0], cloud2[:, 1], cloud2[:, 2], s=1, c=color2, marker='o')

    set_axes_equal(ax)

    ax.view_init(90, -90)
    plt.show()


def show_point_cloud(cloud, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if cloud.shape[1] == 3:
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=1, c='g', marker='o')
    elif cloud.shape[1] == 2:
        z = np.zeros(cloud.shape[0])
        ax.scatter(cloud[:, 0], cloud[:, 1], z, s=1, c='g', marker='o')
    else:
        assert False
    set_axes_equal(ax)

    ax.view_init(90, -90)
    plt.show()


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


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])