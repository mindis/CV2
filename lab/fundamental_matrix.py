import cv2
import sys
import random
import math
import numpy as np
from ransac import ransac
from scipy.spatial import distance
from matplotlib import pyplot as plt


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
        (x1, y1), (x2, y2) = sample
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])

    U, s, V = np.linalg.svd(A)
    smallest_s_index = np.argmin(s)
    F = np.array(V[smallest_s_index]).reshape((3, 3))

    U, s, V = np.linalg.svd(F)
    smallest_s_index = np.argmin(s)
    s[smallest_s_index] = 0

    F = np.dot(U, np.dot(np.diag(s), V))
    return F / F[2, 2]


def normalized_eight_point(points, T1, T2):
    A = []

    for sample in points:
        (x1, y1), (x2, y2) = sample
        x1, y1, _ = np.dot(T1, [x1, y1, 1])
        x2, y2, _ = np.dot(T2, [x2, y2, 1])
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])

    U, s, V = np.linalg.svd(A)
    smallest_s_index = np.argmin(s)
    F = np.array(V[smallest_s_index]).reshape((3, 3))
    U, s, V = np.linalg.svd(F)
    smallest_s_index = np.argmin(s)
    s[smallest_s_index] = 0

    F = np.dot(U, np.dot(np.diag(s), V))
    F = np.dot(T2, np.dot(F, T1))
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


if __name__ == "__main__":
    print
    print 'Calculating epipolar lines', sys.argv[1], 'and', sys.argv[2]
    print '=' * 80

    save_as = None
    if len(sys.argv) > 3:
        save_as = sys.argv[3]

    img1 = cv2.imread(sys.argv[1])[:, :, ::-1]
    img2 = cv2.imread(sys.argv[2])[:, :, ::-1]

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

    matches = filter_matches(matches, 0.75)
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

    F = ransac(sample_pop, algorithm=f,
               error_function=sampson_distance,
               additional_args=additional_args)
    print F

    # TODO: Draw epipolar lines
    n = 2
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    for (i, j) in sample_pop[:n]:
        x = np.dot(F, [i[0], i[1], 1])
        ax1.plot(x[0], x[1], 'r+')
        ax1.plot(i[0], i[1], 'g*')
    ax1.imshow(img1)

    for (i, j) in sample_pop[:n]:
        x = np.dot(F, [j[0], j[1], 1])
        ax2.plot(x[0], x[1], 'r+')
        ax2.plot(j[0], j[1], 'g*')
        # plt.plot([x[0], x[1]], [j[0], j[1]], 'k-')
    ax2.imshow(img2)
    plt.show()