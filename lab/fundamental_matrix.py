import cv2
import sys
import random
import math
import numpy as np
from stitch import filter_matches
from scipy.spatial import distance

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

def eight_point_algorithm(points1, points2, n_sample=8, normalised=True):
    n_sample = min(n_sample, len(points1))

    if normalised:
        T_1 = normalized_t(points1)
        T_2 = normalized_t(points2)
    
    A = []
    
    for points in random.sample(zip(points1, points2), n_sample):
        (x1, y1), (x2, y2) = points
        if normalised:
            x1, y1, _ = np.dot(T_1, [x1, y1, 1])
            x2, y2, _ = np.dot(T_2, [x2, y2, 1])
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])

    U, s, V = np.linalg.svd(A)
    smallest_s_index =  np.argmin(s)
    F = np.array(V[smallest_s_index]).reshape((3, 3))

    if normalised:
        return np.dot(T_2, np.dot(F, T_1))

    U, s, V = np.linalg.svd(F)
    smallest_s_index = np.argmin(s)
    s[smallest_s_index] = 0

    F = np.dot(U, np.dot(np.diag(s), V))
    return F / F[2, 2]

def sampson_distance(coords, coords2, F):
    num = np.dot(np.array(coords2).T, np.dot(F, np.array(coords))) ** 2
    Fp1 = np.dot(F, coords)
    Fp2 = np.dot(F, coords2)
    denum = Fp1[0] ** 2 + Fp1[1] ** 2 + Fp2[0] ** 2 + Fp2[1] ** 2
    return num / denum

def ransac(points1, points2, max_iter=50):
    smallest_err = float('inf')
    best_F = None
    for _ in xrange(max_iter):

        F = eight_point_algorithm(points1, points2)
        err = 0
        for point in zip(points1, points2):
            coords1 = np.append(point[0], 1)
            coords2 = np.append(point[1], 1)
            err += sampson_distance(coords1, coords2, F)

        if err < smallest_err:
            smallest_err = err
            best_F = F
    
    return best_F

def blur_image(img, gaussian_size=(3,3)):
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


    F = ransac(points1, points2, max_iter=1000)
    print F

    #Draw epipolar lines
