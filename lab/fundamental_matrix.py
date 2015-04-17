__author__ = 'finde'

from stitch import *
import random
import math
from operator import itemgetter


def get_normalise_t(points):
    mx = np.mean([x for (x, y) in points])
    my = np.mean([y for (x, y) in points])
    d = np.mean([distance.euclidean(x - mx, y - my) for (x, y) in points])

    T = []
    _s2 = math.sqrt(2)
    T.append([_s2 / d, 0, -mx * _s2 / d])
    T.append([0, _s2 / d, -my * _s2 / d])
    T.append([0, 0, 1])

    return T


def random_subset(iterator, K):
    result = []
    N = 0

    for item in iterator:
        N += 1
        if len(result) < K:
            result.append(item)
        else:
            s = int(random.random() * N)
            if s < K:
                result[s] = item

    return result


def eight_point_algorithm(points1, points2, n_sample=8, normalised=True):
    if normalised:
        T_1 = get_normalise_t(points1)
        T_2 = get_normalise_t(points2)


    # construct matrix A
    matrix = []

    if n_sample > len(points1):
        n_sample = len(points1)

    for random_idx in random_subset(range(len(points1)), n_sample):
        (x1, y1) = points1[random_idx]
        (x2, y2) = points2[random_idx]

        if normalised:
            x1, y1, _ = np.dot(T_1, [x1, y1, 1])
            x2, y2, _ = np.dot(T_2, [x2, y2, 1])
            # print x1, y1, x2, y2

        matrix.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])

    A = np.matrix(matrix)

    U, s, V = np.linalg.svd(A)
    smallest_s_index = min(enumerate(s), key=itemgetter(1))[0]

    F = np.array(V[smallest_s_index]).reshape((3, 3))

    U, s, V = np.linalg.svd(F)
    smallest_s_index = min(enumerate(s), key=itemgetter(1))[0]
    s[smallest_s_index] = 0

    if normalised:
        return np.dot(T_2, np.dot(F, T_1))

    F = np.dot(U, np.dot(np.diag(s), V))
    return F / F[2, 2]


# todo
def sampson_distance(coords, coords2, F):
    num = np.dot(np.array(coords).T, np.dot(F, np.array(coords2))) ** 2

    Fp1 = np.dot(F, coords)
    Fp2 = np.dot(F, coords2)
    denum = Fp1[0] ** 2 + Fp1[1] ** 2 + Fp2[0] ** 2 + Fp2[1] ** 2

    return num / denum


def ransac(matches, img1, kp1, img2, kp2, max_iter=50, THRESHOLD=3):
    _kp1 = []
    _kp2 = []

    for match in matches:
        _kp1.append(kp1[match.queryIdx])
        _kp2.append(kp2[match.trainIdx])

    points1 = np.array([k.pt for k in _kp1])
    points2 = np.array([k.pt for k in _kp2])

    # pick sample --> already sampled randomly from eight_point_algorith
    # run n times and check the agreement
    # best_inliers = 0
    smaller_err = None
    best_F = []
    best_inliers = 0
    for i in range(max_iter):
        F = eight_point_algorithm(points1, points2)

        inliers = 0
        # todo matrix compare
        d = []
        for point in range(len(points1)):

            # transform all key-points of image 1
            coords = np.append(points1[point], 1)
            coords_ = np.append(points2[point], 1)

            d.append(sampson_distance(coords, coords_, F))

        if smaller_err is None or smaller_err > d:
            smaller_err = d
            best_F = F

        # keep transformation matrix if it exceeds the current best
        # if (best_inliers is None) or (best_inliers < inliers):
        #     best_inliers = inliers
        #     best_F = F

    return best_F, best_inliers


if __name__ == "__main__":
    print
    print 'Stitching', sys.argv[1], 'and', sys.argv[2]
    print '=' * 80

    save_as = None
    if len(sys.argv) > 3:
        save_as = sys.argv[3]

    img1 = cv2.imread(sys.argv[1])[:, :, ::-1]
    img2 = cv2.imread(sys.argv[2])[:, :, ::-1]

    # added blur to reduce noise
    img1_ = cv2.GaussianBlur(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    img2_ = cv2.GaussianBlur(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), (3, 3), 0)

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1_, None)
    kp2, des2 = sift.detectAndCompute(img2_, None)

    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})

    matches = matcher.knnMatch(des1, trainDescriptors=des2, k=2)
    print "Match Count:\t\t", len(matches)

    matches = filter_matches(matches, 0.5, sort=True)
    print "Filtered Match Count:\t", len(matches)

    H, status = ransac(matches, img1, kp1, img2, kp2, max_iter=50)
    # # remove the background pair?
    # matched_pair = drawMatches(img1, kp1, img2, kp2, matches[:10])
    # fig = plt.figure(frameon=False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(matched_pair)
    # plt.show()

    # H, status = homography_ransac(matches, 500, kp1, kp2)
    # print "Number of inliers:\t", status
    #
    # canvas = stitch(img2, img1, H)
    # print "Final Image Size: \t", canvas.shape[:2]
    #
    # if save_as:
    # fig = plt.figure(frameon=False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(canvas)
    # fig.savefig(save_as)
    # else:
    # plt.imshow(canvas)
    # plt.show()