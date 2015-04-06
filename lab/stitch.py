# from __future__ import division
import sys
import numpy as np
import cv2
from scipy.spatial import distance
from random import randint

from matplotlib import pyplot as plt


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1, :] = img1
    # Place the next image to the right of it
    out[:rows2, cols1:cols1 + cols2, :] = img2

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def filter_matches(matches, ratio=0.75):
    '''
    reduce weird random matches
    :param matches:
    :param ratio:
    :return:
    '''
    filtered_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            filtered_matches.append(m[0])

    return filtered_matches


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def homography_ransac(matches, n, kp1, kp2, verbose=False):
    THRESHOLD = 10

    best_matrix = None
    best_counter = None
    _kp1 = []
    _kp2 = []

    for match in matches:
        _kp1.append(kp1[match.trainIdx])
        _kp2.append(kp2[match.queryIdx])

    p1 = np.array([k.pt for k in _kp1])
    p2 = np.array([k.pt for k in _kp2])

    for i in range(n):
        # pick n random matches
        counter = 0
        pa = []
        pb = []
        for _ in range(4):
            random_idx = randint(0, len(p1)-1)
            pa.append(p1[random_idx])
            pb.append(p2[random_idx])

        H = find_coeffs(pa, pb)
        H_mat = np.append(H, 1).reshape((3, 3))

        # eval
        for p in range(len(p1)):

            # transform all key-points of image 1
            (x_, y_, w_) = np.dot(H_mat, [p1[p][0], p1[p][1], 1])

            # count inliers
            if distance.euclidean((x_ / w_, y_ / w_), (p2[p][0], p2[p][1])) <= THRESHOLD:
                counter += 1

        # keep transformation matrix if it exceed the current best
        if (best_counter is None) or (best_counter < counter):
            best_counter = counter
            best_matrix = H_mat

        if verbose and i % 100 == 0:
            print i, best_counter, len(matches)

    return best_matrix


img1 = cv2.imread(sys.argv[1])  # queryImage
img2 = cv2.imread(sys.argv[2])  # trainImage

# added blur to reduce noise
img1_ = cv2.GaussianBlur(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (5, 5), 0)
img2_ = cv2.GaussianBlur(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (5, 5), 0)

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# create BFMatcher object => match
# bf = cv2.BFMatcher()
# matches = bf.match(des1, des2)

# create nearest-neighbor matching (alternative)
FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
flann_params = dict(algorithm=FLANN_INDEX_KDTREE,
                    trees=5)
matcher = cv2.FlannBasedMatcher(flann_params, {})
matches = matcher.knnMatch(des2, trainDescriptors=des1, k=2)
print "\t Match Count: ", len(matches)

matches = filter_matches(matches)
print "\t Filtered Match Count: ", len(matches)

# Sort them in the order of their distance.
# matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
# drawMatches(img1, kp1, img2, kp2, matches[:10])

H = homography_ransac(matches, 100, kp1, kp2, verbose=True)

plt.imshow(cv2.warpPerspective(img1, H, img1.shape[:2])[:, :, ::-1])
plt.show()
