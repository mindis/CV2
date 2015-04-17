from __future__ import division
import sys
import numpy as np
import cv2
from scipy.spatial import distance
from random import randint
import math
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

    return out
    # Show the image
    # cv2.imshow('Matched Features', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def filter_matches(matches, ratio=0.75, sort=False):
    '''
    reduce weird random matches
    :param matches:
    :param ratio:
    :return:
    '''
    filtered_matches = []

    if sort:
        matches = sorted(matches, key=lambda x: x[0].distance)

    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            filtered_matches.append(m[0])

    return filtered_matches


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix)
    B = np.array(pb).reshape(8)

    H = np.dot(np.linalg.pinv(A), B)
    H = np.array(H).reshape(8)
    H = np.concatenate((H, np.array([1]))).reshape((3, 3))
    return H


def homography_ransac(matches, n, kp1, kp2, verbose=False, local_optimisation=False):
    """
    Find homography matrix from 2 images
    :param matches: pair keypoints matches between image 1 and image 2
    :param n: number of ransac iteration
    :param kp1: list of image 1's keypoints
    :param kp2: list of image 2's keypoints
    :param verbose: print debug code
    :param local_optimisation: using LO-Ransac
    :return: Homograpy matrix, number of inliers
    """

    THRESHOLD = 3

    best_H = None
    best_inliers = None
    _kp1 = []
    _kp2 = []

    for match in matches:
        _kp1.append(kp1[match.queryIdx])
        _kp2.append(kp2[match.trainIdx])

    points1 = np.array([k.pt for k in _kp1])
    points2 = np.array([k.pt for k in _kp2])

    for i in range(n):
        # pick n random matches
        inliers = 0
        picked1 = []
        picked2 = []
        for _ in range(4):
            random_idx = randint(0, len(points1) - 1)
            picked1.append(points1[random_idx])
            picked2.append(points2[random_idx])

        H = find_coeffs(picked1, picked2)
        # eval
        for point in range(len(points1)):

            # transform all key-points of image 1
            coords = np.append(points1[point], 1)
            (x, y, w) = np.dot(H, coords)

            # count inliers
            if distance.euclidean((x / w, y / w), points2[point]) <= THRESHOLD:
                inliers += 1

        # keep transformation matrix if it exceeds the current best
        if (best_inliers is None) or (best_inliers < inliers):
            if local_optimisation:
                # todo ::
                pass
            else:
                best_inliers = inliers
                best_H = H

    return best_H, best_inliers


def findDimensions(image, H):
    h, w = image.shape[:2]  # Height, width
    base = [[0, 0, h, h],
            [0, w, 0, w],
            [1, 1, 1, 1]]

    base = np.array(base)
    base_prime = H.dot(base)
    new_coords = base_prime[:2] / base_prime[2]
    min_x = min(new_coords[0])
    min_y = min(new_coords[1])
    max_x = max(new_coords[0])
    max_y = max(new_coords[1])

    return min_x, max_x, min_y, max_y


def create_mask(img):
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY_INV)
    return mask


def remove_black_edges(img):
    # Crop off the black edges
    final_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    best_rect = (0, 0, 0, 0)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        deltaHeight = h - y
        deltaWidth = w - x

        area = deltaHeight * deltaWidth

        if area > max_area and deltaHeight > 0 and deltaWidth > 0:
            max_area = area
            best_rect = (x, y, w, h)

    if max_area > 0:
        img_crop = img[best_rect[1]:best_rect[1] + best_rect[3],
                   best_rect[0]:best_rect[0] + best_rect[2]]

        return img_crop

    return img


def stitch(new_img, base_img, H):
    """
    Stitch 2 images together
    :param new_img: image to stitch
    :param base_img: base image
    :param H: Homography coordinate
    :param BASE_ON_TOP: Set to True for placing new image at the back
    :return: stitched image
    """
    # do the invert because we want to transform new_image to the base_image
    H_inv = np.linalg.pinv(H)

    min_x, max_x, min_y, max_y = findDimensions(new_img, H_inv)

    # Adjust max_x and max_y by base img size
    max_x = max(max_x, base_img.shape[1])
    max_y = max(max_y, base_img.shape[0])

    # create translation transform matrix
    move_h = np.matrix(np.identity(3), np.float32)

    if min_x < 0:
        move_h[0, 2] -= math.ceil(min_x)
        max_x -= min_x

    if min_y < 0:
        move_h[1, 2] -= math.ceil(min_y)
        max_y -= min_y

    mod_inv_h = move_h * H_inv

    img_w = int(math.ceil(max_x))
    img_h = int(math.ceil(max_y))

    base_img_warp = cv2.warpPerspective(base_img, move_h, (img_w, img_h))
    new_img_warp = cv2.warpPerspective(new_img, mod_inv_h, (img_w, img_h))

    # create empty matrix
    canvas = np.zeros((img_h, img_w, 3), np.uint8)

    # combining
    canvas = cv2.max(canvas, base_img_warp)
    canvas = cv2.max(canvas, new_img_warp)

    # remove black edges
    canvas = remove_black_edges(canvas)

    return canvas


if __name__ == '__main__':
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

    matches = filter_matches(matches)
    print "Filtered Match Count:\t", len(matches)

    # plt.imshow(drawMatches(img1, kp1, img2, kp2, matches))
    # plt.show()

    H, status = homography_ransac(matches, 500, kp1, kp2)
    print "Number of inliers:\t", status

    canvas = stitch(img2, img1, H)
    print "Final Image Size: \t", canvas.shape[:2]

    if save_as:
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(canvas)
        fig.savefig(save_as)
    else:
        plt.imshow(canvas)
        plt.show()