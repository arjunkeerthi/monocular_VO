"""
Credit goes to Avi Singh: https://github.com/avisingh599/mono-vo.
The code below The code below builds of his C++ code, with modifications..

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import cv2
import math
import time
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
orb = cv2.ORB_create(nfeatures=3000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
enable_nms = True

# Change sequence value to run on different image sets.
# Report used 0, 2, 5, and 9.
sequence = 9

# Path to directory that contains the data_odometry_poses and
# data_odometry_color folders from the Kitti Benchmark data set.
# <add the correct path here for the computer being used>
path = ""

fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
gftt = cv2.GFTTDetector_create(maxCorners=3000)
akaze = cv2.AKAZE_create()
agast = cv2.AgastFeatureDetector_create()

detector_color_dict = {
    "FAST": (102, 252, 255),
    "GFTT": (0, 0, 255),
    "AGAST": (255, 179, 102)
}


# Read lines from ground truth pose text file and return as array.
# Format for each line of file is the 3x4 projection matrix flattened into
# a list of 12 numbers (projection matrix is [R | t], where R is 3x3 rotation
# matrix and t is 3x1 translation vector. After testing this out, it appears
# that each line gives these matrices with respect to the starting pose (no need
# to chain every prior together to get current position). So, first 3 numbers is
# first row of R, 4th number is 1st element of t. 5th-7th give second row of R,
# 8th gives second element of t. And finally 9th-11th give third row of R, 12th
# gives third element of t.
def getPoses():
    def parse_line(line):
        parts = line.split(' ')
        return list(map(lambda p: float(p), parts))

    with open(f"{path}\\data_odometry_poses\\dataset\\poses\\"
              f"{sequence:02}.txt", 'r') as f:
        lines = f.readlines()
        return list(map(parse_line, lines))


# Compute distance between two poses to use as scale (as translation from
# recoverPose() should be of unit length).
def getAbsoluteScale(pose1, pose2):
    x1 = pose1[3]
    y1 = pose1[7]
    z1 = pose1[11]
    x2 = pose2[3]
    y2 = pose2[7]
    z2 = pose2[11]
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)


# Extract rotation and translation matrices from ground truth pose.
def getTransformation(pose):
    rotation = np.array([pose[:3], pose[4:7], pose[8:11]])
    translation = np.array([[pose[3]], [pose[7]], [pose[11]]])
    return rotation, translation


# Extract FAST features from image and convert to a numpy array of points.
# Returned array has shape Nx2, where N is the number of features. x is first col,
# y is second col.
def featureDetection(img, type="FAST"):
    keypoints = None
    if type == "FAST":
        keypoints = akaze.detect(img)
    elif type == "GFTT":
        keypoints = gftt.detect(img)
    elif type == "AKAZE":
        keypoints = akaze.detect(img)
    elif type == "AGAST":
        keypoints = agast.detect(img)

    return cv2.KeyPoint_convert(keypoints)


# Track features provided in points2 from img1 to img2, removing the features
# from points1 and points2 that are not good.
def featureTracking(img1, img2, points1):
    lk_params = dict(winSize  = (21,21),
                     maxLevel = 3,
                     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
                                 30, 0.001))

    points2, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, points1, None,
                                                    **lk_params)

    # Mask to retain features with status of 1 and have nonnegative x and y
    # position.
    status_mask = np.squeeze(status == 1)
    points1_mask = (points1[:, 0] >= 0) & (points1[:, 1] >= 0)
    points2_mask = (points2[:, 0] >= 0) & (points2[:, 1] >= 0)
    keep_mask = status_mask & points1_mask & points2_mask

    points1 = points1[keep_mask, :]
    points2 = points2[keep_mask, :]

    return points1, points2


# Get image from Kitti odometry dataset with provided index and from left rgb camera.
# Return both colored and grayscale images.
def getImage(index):
    img_path = f"{path}\\data_odometry_color\\dataset\\sequences\\" \
        f"{sequence:02}\\image_2\\{index:06}.png"
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray


# Run main odometry on Kitti images.
def main():
    # Get all the ground truth poses
    poses = getPoses()

    # Detector type.
    # Change type here for each run.
    # Available types: "FAST", "GFTT", "AGAST".
    type = "FAST"

    # Get first two images
    index = 0
    img1_c, img1 = getImage(index)
    index += 1
    img2_c, img2 = getImage(index)
    index += 1

    # Detect features in img1 and track to img2
    points1 = featureDetection(img1, type)
    points1, points2 = featureTracking(img1, img2, points1)

    # Camera matrix extracted from calib.txt for sequence 00.
    # Format of this file found from this answer:
    # https://stackoverflow.com/a/50211379.
    # To summarize: 5 lines, first four define projection matrix for each of
    # the 4 cameras: 1st - left grayscale, 2nd - right grayscale, 3rd - left rgb,
    # 4th - right rgb. I am using the left camera images, so I'm using the 3rd
    # camera. Each line 12 numbers, defining 3x4 projection matrix flattened:
    # [C | t], where C is the camera matrix. So I just copied those numbers over
    # to here.
    camera_matrix = np.array([
        [7.18856e+02, 0.0, 6.071928e+02],
        [0.0, 7.18856e+02, 1.852157e+02],
        [0.0, 0.0, 1.0]
    ])

    # Compute essential matrix between first two images and then recover rotation
    # and translation matrices between the two images.
    E, mask = cv2.findEssentialMat(points2, points1, camera_matrix, cv2.RANSAC,
                                   0.999)
    retval, R, t, mask = cv2.recoverPose(E, points2, points1, camera_matrix)

    prev_img = img2
    prev_features = points2
    R_f = R
    t_f = t

    # Will store plot of vehicle's trajectory.
    # Use first line when running on a sequence for the first time to initialize
    # trajectory image.
    # Use second line to load trajectory image and draw current feature's trajectory
    # over it.
    trajectory = np.ones((800, 800, 3), dtype=np.uint8)  # Equivalent to cv2.CV_8UC3
    # trajectory = cv2.imread(f"trajectory{sequence:02}.png")

    cv2.namedWindow("Road facing camera")
    cv2.namedWindow("Trajectory")

    total_error = 0
    start = time.time()
    for i in range(2, 1502):
        # Get ground truth rotation and translation.
        R_gt, t_gt = getTransformation(poses[index])

        # Get current image.
        curr_img_c, curr_img = getImage(index)
        index += 1

        # Track features from prev to curr image.
        prev_features, curr_features = featureTracking(prev_img, curr_img,
                                                       prev_features)

        # Compute essential matrix and compute rotation and translation from prev
        # to curr.
        E, mask = cv2.findEssentialMat(curr_features, prev_features, camera_matrix,
                                       cv2.RANSAC, 0.999)
        retval, R, t, mask = cv2.recoverPose(E, curr_features, prev_features,
                                             camera_matrix)

        # Calculate scale and compute the new translation and rotation.
        scale = getAbsoluteScale(poses[index-1], poses[index])
        if scale > 0.1 and t[2] > t[0] and t[2] > t[1]:
            t_f = t_f + scale * (R_f @ t)
            R_f = R_f @ R

        # If the number of features being tracked drops below 2000, re-detect
        # features.
        if len(prev_features) < 2000:
            prev_features = featureDetection(prev_img, type)
            prev_features, curr_features = featureTracking(prev_img, curr_img,
                                                           prev_features)

        prev_img = curr_img
        prev_features = curr_features

        total_error += np.linalg.norm(t_f - t_gt)

        # Compute current location.
        # Admittedly this part confuses me a little. It looks like t_f, R_f are
        # the matrices from starting point to current point. t_f is the translation
        # from start to current, so I think that's why we index this directly to find
        # (x,y) coordinate. Orientation doesn't matter for plot, so we don't consider
        # R here (if we needed a bearing for the car, then we probably would have to
        # use R as well).
        x = int(t_f[0][0]) + 200
        y = 800 - (int(t_f[2][0]) + 200)
        cv2.circle(trajectory, (x,y), 1, detector_color_dict[type], 2)

        # Ground truth location.
        # Uncomment this line after the first run on a sequence to avoid re-drawing
        # the ground truth poses every time.
        x_gt = int(t_gt[0][0]) + 200
        y_gt = 800 - (int(t_gt[2][0]) + 200)
        cv2.circle(trajectory, (x_gt,y_gt), 1, (0,255,0), 2)

        # Display text for current position.
        # Uncomment to see position for debugging purposes.
        # cv2.rectangle(trajectory, (10,30), (550,50), (0,0,0), cv2.FILLED)
        # text = f"Coordinates: x = {t_f[0][0]:.4f}m, y = {t_f[1][0]:.4f}m, " \
        #     f"z = {t_f[2][0]:.4f}m"
        # cv2.putText(trajectory, text, (10,50), cv2.FONT_HERSHEY_PLAIN, 1,
        #             (255,255,255), 1, cv2.LINE_8)

        cv2.imshow("Road facing camera", curr_img_c)
        cv2.imshow("Trajectory", trajectory)
        cv2.waitKey(1)

    end = time.time()
    cv2.imwrite(f"trajectory{sequence:02}.png", trajectory)
    cv2.destroyAllWindows()
    print(f"Total error for {type}: {total_error:.4f}")
    print(f"Total time: {end-start} seconds")
    print(f"Average fps: {1500/(end-start):.4f}")

    fig, ax = plt.subplots()
    extent = (-250, 550, -200, 600)
    rgb_trajectory = cv2.cvtColor(trajectory, cv2.COLOR_BGR2RGB)
    im = ax.imshow(rgb_trajectory, origin='upper', extent=extent)
    plt.show()


if __name__ == "__main__":
    main()
