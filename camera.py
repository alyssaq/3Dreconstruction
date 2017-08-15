import numpy as np
import processor
import transformers


class Camera(object):
    """ Class for representing pin-hole camera """

    def __init__(self, P=None, K=None, R=None, t=None):
        """ P = K[R|t] camera model. (3 x 4)
         Must either supply P or K, R, t """
        if P is None:
            try:
                self.extrinsic = np.hstack([R, t])
                P = np.dot(K, self.extrinsic)
            except TypeError as e:
                print('Invalid parameters to Camera. Must either supply P or K, R, t')
                raise

        self.P = P     # camera matrix
        self.K = K     # intrinsic matrix
        self.R = R     # rotation
        self.t = t     # translation
        self.c = None  # camera center

    def project(self, X):
        """ Project 3D homogenous points X (4 * n) and normalize coordinates.
            Return projected 2D points (2 x n coordinates) """
        x = np.dot(self.P, X)
        x[0, :] /= x[2, :]
        x[1, :] /= x[2, :]

        return x[:2, :]

    def qr_to_rq_decomposition(self):
        """ Convert QR to RQ decomposition with numpy.
        Note that this could be done by passing in a square matrix with scipy:
        K, R = scipy.linalg.rq(self.P[:, :3]) """
        Q, R = np.linalg.qr(np.flipud(self.P).T)
        R = np.flipud(R.T)
        return R[:, ::-1], Q.T[::-1, :]

    def factor(self):
        """ Factorize the camera matrix P into K,R,t with P = K[R|t]
          using RQ-factorization """
        if self.K is not None and self.R is not None:
            return self.K, self.R, self.t  # Already been factorized or supplied

        K, R = self.qr_to_rq_decomposition()
        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if np.linalg.det(T) < 0:
            T[1, 1] *= -1

        self.K = np.dot(K, T)
        self.R = np.dot(T, R)  # T is its own inverse
        self.t = np.dot(np.linalg.inv(self.K), self.P[:, 3])

        return self.K, self.R, self.t

    def center(self):
        """  Compute and return the camera center. """
        if self.c is not None:
            return self.c
        elif self.R:
            # compute c by factoring
            self.c = -np.dot(self.R.T, self.t)
        else:
            # P = [M|−MC]
            self.c = np.dot(-np.linalg.inv(self.c[:, :3]), self.c[:, -1])
        return self.c


def test():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import cv2

    # load points
    points = processor.read_matrix('testsets/house.p3d').T  # 3 x n
    points = processor.cart2hom(points)  # 4 x n

    img = cv2.imread('testsets/house1.jpg')
    height, width, ch = img.shape

    K = np.array([
        [width * 30, 0, width / 2],
        [0, height * 30, height / 2],
        [0, 0, 1]])
    R = np.array([  # No rotation
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    t = np.array([[0], [0], [100]])  # t != 0 to be away from image plane

    # Setup cameras
    cam = Camera(K=K, R=R, t=t)
    x = cam.project(points)

    rotation_angle = 20
    rotation_mat = transformers.rotation_3d_from_angles(rotation_angle)
    cam = Camera(K=K, R=rotation_mat, t=t)
    x2 = cam.project(points)

    # Plot actual 3d points
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.plot(points[0], points[1], points[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=140, azim=0)

    # Plot 3d to 2d projection
    f, ax = plt.subplots(2, sharex=True, sharey=True)
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.99,
                        top=0.95, wspace=0, hspace=0.01)
    ax[0].set_aspect('equal')
    ax[0].set_title(
        '3D to 2D projection. Bottom x-axis rotated by {0}°'.format(rotation_angle))
    ax[0].plot(x[0], x[1], 'k.')
    ax[1].plot(x2[0], x2[1], 'k.')
    plt.show()


if __name__ == '__main__':
    test()
