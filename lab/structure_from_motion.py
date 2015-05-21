import cv2
import sys
import random
import math
import numpy as np
from ransac import ransac
from scipy.spatial import distance
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def filter_matches(matches, ratio=0.75):
    filtered_matches = []
    for m in matches:
        if m[0].distance < m[1].distance * ratio:
            filtered_matches.append(m[0])
    return filtered_matches


def normalized_t(points):
    mean_x = np.mean([x for (x, y) in points])
    mean_y = np.mean([y for (x, y) in points])
    mean_distance = np.mean([distance.euclidean(x - mean_x, y - mean_y) for (x, y) in points])
    T = []
    square_2 = math.sqrt(2)
    T.append([square_2 / mean_distance, 0, -mean_x * square_2 / mean_distance])
    T.append([0, square_2 / mean_distance, -mean_y * square_2 / mean_distance])
    T.append([0, 0, 1])
    return T


def eight_point(points, **kwargs):
    A = []

    for sample in points:
        (x, y), (xp, yp) = sample
        A.append([xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1])

    U, s, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    U, s, V = np.linalg.svd(F)
    s[2] = 0
    F = np.dot(U, np.dot(np.diag(s), V))

    return F / F[2, 2]


def normalized_eight_point(points, T1, T2):
    normalized_points = []
    for sample in points:
        (x1, y1), (x2, y2) = sample
        x1, y1, _ = np.dot(T1, [x1, y1, 1])
        x2, y2, _ = np.dot(T2, [x2, y2, 1])
        normalized_points.append(((x1,y1),(x2, y2)))
    
    F = eight_point(normalized_points)
    F = np.dot(np.array(T1).T, np.dot(F, np.array(T2)))
    return F / F[2, 2]


def sampson_distance(points, F):
    coords1, coords2 = points
    coords1 = [coords1[0], coords1[1], 1]
    coords2 = [coords2[0], coords2[1], 1]

    num = np.dot(np.array(coords2).T, np.dot(F, np.array(coords1))) ** 2
    Fp1 = np.dot(F, coords1)
    Fp2 = np.dot(F, coords2)
    denum = (Fp1 ** 2).sum() + (Fp2 ** 2).sum()
    return num / denum


def blur_image(img, gaussian_size=(3, 3)):
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), gaussian_size, 0)


def plot_epipolar_line(im, F, x):
    """ 
        Plot the epipole and epipolar line F*x=0
        in an image. F is the fundamental matrix 
        and x a point in the other image.
    """

    m, n = im.shape[0], im.shape[1]
    line = np.dot(F, x)

    # epipolar line parameter and values
    t = np.linspace(0, n, 100)
    lt = np.array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

    plt.plot(t, lt, linewidth=2)


def get_fundamental_matrix(img1, img2, filter_match=.5, n_iter=500, acceptance=0.5):
    """
        Gets the fundamental matrix for `img1` and `img2` using RANSAC
    """
    # added blur to reduce noise
    img1_ = blur_image(img1)
    img2_ = blur_image(img2)

    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(img1_, None)
    kp2, des2 = sift.detectAndCompute(img2_, None)

    FLANN_INDEX_KDTREE = 1
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})

    matches = matcher.knnMatch(des1, trainDescriptors=des2, k=2)
    print "Match Count:\t\t", len(matches)

    matches = filter_matches(matches, filter_match)
    print "Filtered Match Count:\t", len(matches)

    points1, points2 = [], []

    for match in matches:
        points1.append(kp1[match.queryIdx].pt)
        points2.append(kp2[match.trainIdx].pt)

    points1 = np.array(points1)
    points2 = np.array(points2)

    sample_pop = zip(points1, points2)
    f = eight_point
    normalize = True
    additional_args = {}

    if normalize:
        T1 = normalized_t(points1)
        T2 = normalized_t(points2)
        additional_args['T1'] = T1
        additional_args['T2'] = T2
        f = normalized_eight_point

    F, best_sample = ransac(sample_pop, algorithm=f,
                            error_function=sampson_distance,
                            additional_args=additional_args,
                            acceptance=acceptance,
                            n_iter=n_iter)

    return F, best_sample


def draw_epipolar_line(img1, img2, F, best_sample):
    """
        Draws the epipolar lines for the best F and the best sample
        returned by RANSAC for `img1` and `img2`
    """
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img1)
    ax1.set_xlim([img1.shape[1], 0])
    ax1.set_ylim([img1.shape[0], 0])

    ax2.imshow(img2)
    ax2.set_xlim([img2.shape[1], 0])
    ax2.set_ylim([img2.shape[0], 0])

    for (point1, point2) in best_sample:
        plt.sca(ax1)
        point = [point1[0], point1[1], 1]
        plt.plot(point1[0], point1[1], '*')
        plot_epipolar_line(img1, F.T, point)

        plt.sca(ax2)
        point = [point2[0], point2[1], 1]
        plt.plot(point2[0], point2[1], '*')
        plot_epipolar_line(img2, F, point)

    plt.show()


def generate_point_view_matrix(dirname, use_cache=True):
    # for all images in directory, create a sequential pair
    point_view_matrix = pd.DataFrame()

    if use_cache and isfile(dirname + '.csv'):
        point_view_matrix = point_view_matrix.from_csv(open(dirname + '.csv', 'r'))
    else:
        images = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        # images = images[:5] + images[-5:]

        for i in xrange(len(images)):
            i2 = i + 1 if i + 1 < len(images) else 0
            print
            print 'Calculating epipolar lines', images[i], 'and', images[i2]
            print '=' * 80

            img1 = cv2.imread(dirname + '/' + images[i])[:, :, ::-1]
            img2 = cv2.imread(dirname + '/' + images[i2])[:, :, ::-1]

            F, inliers = get_fundamental_matrix_for_pair_image(img1, img2, filter_match=.75, n_iter=50, acceptance=0.01)
            if inliers is None:
                continue

            if point_view_matrix.shape[1] == 0:
                for iix, (point1, point2) in enumerate(inliers):
                    fp = 'fp%d' % (iix + 1)
                    point_view_matrix.ix['img%d_x' % i, fp] = point1[0]
                    point_view_matrix.ix['img%d_y' % i, fp] = point1[1]
                    point_view_matrix.ix['img%d_x' % i2, fp] = point2[0]
                    point_view_matrix.ix['img%d_y' % i2, fp] = point2[1]
            else:
                for (point1, point2) in inliers:
                    try:
                        point_view_matrix.loc['img%d_x' % i]
                    except:  # If the previous image had no matched points
                        fp = 'fp%d' % (point_view_matrix.shape[1] + 1)
                        point_view_matrix.ix['img%d_x' % i2, fp] = point2[0]
                        point_view_matrix.ix['img%d_y' % i2, fp] = point2[1]
                        continue

                    for fp, p in enumerate(
                            zip(point_view_matrix.loc['img%d_x' % i], point_view_matrix.loc['img%d_y' % i])):
                        if (point1[0], point1[1]) == p:
                            point_view_matrix.ix['img%d_x' % i2, fp] = point2[0]
                            point_view_matrix.ix['img%d_y' % i2, fp] = point2[1]
                            break
                    else:  # If the point was not found
                        fp = 'fp%d' % (point_view_matrix.shape[1] + 1)
                        point_view_matrix.ix['img%d_x' % i, fp] = point1[0]
                        point_view_matrix.ix['img%d_y' % i, fp] = point1[1]
                        point_view_matrix.ix['img%d_x' % i2, fp] = point2[0]
                        point_view_matrix.ix['img%d_y' % i2, fp] = point2[1]

        point_view_matrix.to_csv(open(dirname + '.csv', 'w'))

    return point_view_matrix.as_matrix()


def move_to_mean(pv_matrix):
    """
        Moves the center of every image to the centroid of the image
    """
    return (pv_matrix.T - np.nanmean(pv_matrix, axis=1)).T

def get_dense_submatrix(pv_matrix, offset = 0):
    """
        Normalize the point coordinates by translating them to the mean of the points in each view
    """
    return (pv_matrix.T - np.nanmean(pv_matrix, axis=1)).T


def get_dense_submatrix(pv_matrix):
    """
    Finds the dense submatrix involving the first column in the point view 
    matrix. Then removes those points from the point view matrix aswell.
    """
    
    #See in what photos the first point is visible
    if not pv_matrix.shape[1]:
        return np.array([]), np.array([])
    
    col = pv_matrix[:,0]
    bool_col = ~np.isnan(col)

    #Find out which points at least overlap
    rel_part = pv_matrix[bool_col, :]
    overlapping_columns = (~np.isnan(rel_part)).all(axis=0)
    if not overlapping_columns.shape[0] >= 3:
        return get_dense_submatrix(pv_matrix[:,1])

    #Change PVM to not overlap and dense submatrix to do overlap
    pvm = pv_matrix[:, ~overlapping_columns]
    dense = rel_part[:, overlapping_columns]
    
    return dense, pvm

def eliminate_ambiguity(motion, structure):
    """
    Eliminates the affine ambiguity of the motion and structure matrices.
    """
    
    n_camera = int(motion.shape[0])/2
    A = motion
    B = np.empty((n_camera * 2,3))
    for i in range(n_camera):
        B[(i*2) : (i*2)+2, :] = np.linalg.pinv(A[(i*2) : (i*2)+2,:].T)
    
    L = np.linalg.lstsq(A,B)[0]
    C = np.linalg.cholesky(L)

    motion = np.dot(motion, C)
    structure = np.dot(np.linalg.pinv(C), structure)
    
    return motion, structure

def structure_motion_from_PVM(PVM):
    PVM = move_to_mean(PVM)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    
    while PVM.shape[0] > 3:
        dense, PVM = get_dense_submatrix(PVM)
        if not dense.shape[0] > 3:
            continue

        U, s, V = np.linalg.svd(dense)
        U3 = U[:,:3]
        S3 = np.diag(s[:3])
        V3 = V[:3]

        motion = U3    
        structure = np.dot(S3, V3)

        if structure.shape[0] < 3:
            break

        motion, structure = eliminate_ambiguity(motion, structure)

        xs = list(structure[0])
        ys = list(structure[1])
        zs = list(structure[2])

        ax.scatter(xs, ys, zs)

        plt.show()
        fig.clf()
        fig = plt.figure()
        ax = fig.gca(projection='3d')



if __name__ == "__main__":
    PVM = generate_point_view_matrix(sys.argv[1])
    #plt.matshow(PVM, interpolation='nearest', cmap='Greys')
    #plt.show()
    structure_motion_from_PVM(PVM)