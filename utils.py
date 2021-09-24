import numpy as np
import math


def is_rotation_matrix(r):
    """Check if a matrix is a valid rotation matrix.
    referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    rt = np.transpose(r)
    should_be_identity = np.dot(rt, r)
    i = np.identity(3, dtype=r.dtype)
    n = np.linalg.norm(i - should_be_identity)
    return n < 1e-6


def rotation2euler(r):
    """Convert rotation matrix to euler angles.
    referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles
    """
    assert (is_rotation_matrix(r))
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def euler2rotation(theta):
    """Convert euler angles to rotation matrix.
    referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    return np.dot(R_z, np.dot(R_y, R_x))


def trans_n_rot(Rt):
    Rt = np.reshape(np.array(Rt), (3, 4))
    t = Rt[:, -1]
    R = Rt[:, :3]

    assert (is_rotation_matrix(R))
    pose = np.concatenate((t, R.flatten()))
    assert (pose.shape == (12,))
    return pose


def create_pose_data(pose_dir, seq):
    """
    load ground truth pose data from txt
    pose_dir: path to pose folder, string
    seq: video sequence, string
    return: numpy array of poses, (n, 12)
    """
    fn = '{}{}.txt'.format(pose_dir, seq)
    with open(fn) as f:
        lines = [line.split('\n')[0] for line in f.readlines()]
        poses = [trans_n_rot([float(value) for value in l.split(' ')]) for l in lines]
        poses = np.array(poses)  # (n, 12) [x, y, z, rotation]
    return poses


def cal_rel_pose(R1, R2, t1, t2):
    """
    calculate relative pose
    R1, R2: rotation matrix, (3, 3) np array
    t1, t2: translation, (3, ) np array
    return: euler angle and translation
    """
    R = R1.dot(R2)
    t = R1.dot(t2 - t1)
    euler = rotation2euler(R)
    return euler, t


class EarlyStopping(object):
    """
    referred from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    """
    def __init__(self, mode='min', min_delta=0, patience=25, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def adjust_lr(ep, base_lr, optimizer, gamma=0.5, decay_per_ep=20):
    decay_rate = int(math.floor(ep / decay_per_ep))
    decay_rate = gamma ** decay_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * decay_rate


THRESHOLD = 5e-7

# numpy implementation
def skew_matrix(x):
    """
    generate skew matrix. R^3 to 3x3 matrix
    :param x: R^3 vector. size = (3,)
    :return: 3x3 matrix

    skew matrix of (x, y, z):
            0    -z   y
            z    0   -x
            -y   x    0
     """
    skew = np.zeros((3, 3))
    skew[0, 1] = -x[2]
    skew[1, 0] = x[2]
    skew[0, 2] = x[1]
    skew[2, 0] = -x[1]
    skew[1, 2] = -x[0]
    skew[2, 1] = x[0]
    return skew


def calculateRV(w):
    """
    w: R^3
    return: R - rotation, V - for calculating translation

    exponetial map:
    theta = root(w.T w)
    A = (sin theta)/theta
    B = (1 - cos theta)/theta^2
    C = (1 - A)/theta^2 = (theta - sin theta)/theta^3
    R = I + A*skew + B*skew^2
    V = I + B*skew + C*skew^2
    t = Vu
    """
    # generate skew matrix
    skew = skew_matrix(w)
    skew_sqr = np.dot(skew, skew)

    theta_sqr = np.dot(w.T, w)
    theta = np.sqrt(theta_sqr)  # scalar
    theta_cube = theta_sqr * theta
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # a, b, c are scalar
    # if theta square is small, use taylor expansions of b, c instead of closed-form representation
    if theta_sqr < THRESHOLD:
        a = 1 - theta_sqr / 6 + theta_sqr ** 2 / 120 - theta_sqr ** 3 / 5040 + theta_sqr ** 4 / 362880
        b = 0.5 - theta_sqr / 24 + theta_sqr ** 2 / 720 + theta_sqr ** 3 / 40320 + theta_sqr ** 4 / 362880
        c = 1 / 6 - theta_sqr / 120 + theta_sqr ** 2 / 5040 - theta_sqr ** 3 / 362880 + theta_sqr ** 4 / 39916800
    else:
        a = sin_theta / theta
        b = (1 - cos_theta) / theta_sqr
        c = (theta - sin_theta) / theta_cube

    # calculate R and V
    I = np.eye(3)
    R = I + a * skew + b * skew_sqr
    V = I + b * skew + c * skew_sqr
    return R, V


def exponential_map_se3(wu):
    """
    expoential map of se(3) -> SE(3)
    :param wt: 6d coordinate in se(3), wt = (w, t)
    :return: SE(3) transformation matrix
    """
    w = wu[:3]
    u = wu[3:]
    R, V = calculateRV(w)
    transformation = np.eye(4)
    transformation[:3, :3] = R
    t = np.dot(V, u)
    transformation[:3, 3] = t
    return transformation


def exponential_map_so3(w):
    """
    exponential map of so(3)
    :param w: 3d coordinate in so(3)
    :return: SO(3) rotation matrix
    """
    return calculateRV(w)[0]


def log_SO3(r):
    """
    log function on SO(3)
    SO(3) -> so(3)
    :param r: rotation matrix, size = (3, 3)
    :return: skew matrix, size = (3, 3); theta

    x = (tr(R) - 1)/2
    theta = acos(x)
    ln(R) = 1/2 * theta/sin(theta) * (R - R.T)
    """
    assert is_rotation_matrix(r), 'invalid rotation matrix'
    cos_theta = np.clip(0.5 * (np.trace(r) - 1), -1., 1.)  # make sure cos(theta) is between -1 and 1
    theta = np.arccos(cos_theta)

    # ln(R)
    if theta <= THRESHOLD:
        c = 0.5 + theta ** 2 / 12 + theta ** 4 * 7 / 720 + theta ** 6 * 31 / 30240 + theta ** 8 * 127 /1209600
    else:
        c = 0.5 * theta / np.sin(theta)
    skew = c * (r - r.T)  # size = (3, 3)
    return skew, theta


def log_SE3_2_R6(rt):
    """
    SE(3) -> se(3) -> R^6
    :param rt: transformation matrix, size = (3, 4) or (4, 4) if homogeneous
    :return: 6d vector representation

    u = inverse(V)t
    inverse(V) = I - 0.5 * skew + 1/theta^2 * (1 - A/2B) * skew^2
    """
    r = rt[:3, :3]
    t = rt[:3, 3]

    skew, theta = log_SO3(r)
    skew_sqr = np.dot(skew, skew)
    theta_sqr = theta ** 2

    if theta_sqr <= THRESHOLD:
        c = theta_sqr / 12 + theta_sqr ** 2 / 720 + theta_sqr ** 3 / 30240 + theta_sqr ** 4 / 1209600
        # c = 1 / 12 + theta_sqr / 720 + theta_sqr ** 2 / 30240 + theta_sqr ** 3 / 1209600 + theta_sqr ** 4 / 47900160
    else:
        a = np.sin(theta) / theta
        b = (1 - np.cos(theta)) / theta_sqr
        c = (1 - (a / (2 * b))) / theta_sqr
        # c = (2 * np.sin(theta/2) - theta * np.cos(theta/2)) / (2 * theta ** 2 * np.sin(theta/2))

    # V inverse
    I = np.eye(3)
    v_i = I - 0.5 * skew + c * skew_sqr

    u = np.dot(v_i, t)  # size = (b_size, 3, 1)
    w = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return w, u


