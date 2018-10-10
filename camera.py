import numpy as np
import cv2
import glob

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

WIDTH = 6
HEIGHT = 9

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.JPG')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:HEIGHT,0:WIDTH].T.reshape(-1,2)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret = False
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (HEIGHT,WIDTH), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corner2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corner2)
try:
    # Camera Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Printing the calibration output
    print('Camera Matrix:\n', mtx)
    print('Distortion:\n', dist)
    print('Rotation Vector :\n', rvecs)
    print('Translation Vector:\n', tvecs)

    # Undistortion
    img = cv2.imread('camera5.JPG')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite('Calibresult.jpg', dst)

except:
    print("Exception Occured")
pass

def compute_homography(fp,tp):

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # create matrix for linear method, 2 rows for each correspondence pair
    num_corners = fp.shape[0]

    # construct constraint matrix
    A = np.zeros((num_corners * 6, 9))
    A[0::2, 0:3] = fp
    A[1::2, 3:6] = fp
    A[0::2, 6:9] = fp * -np.repeat(np.expand_dims(tp[:, 0], axis=1), 3, axis=1)
    A[1::2, 6:9] = fp * -np.repeat(np.expand_dims(tp[:, 1], axis=1), 3, axis=1)

    # solve using *naive* eigenvalue approach
    D, V = np.linalg.eig(A.transpose().dot(A))

    H = V[:, np.argmin(D)].reshape((3, 3))

    cv2.findHomography(img, dst, H)

    # normalise and return
    return H


# Webcam capture
cap = cv2.VideoCapture(0)

images = glob.glob('AR.JPG')  # The .jpg images in the folder can be displayed
currentImage = 0  # the first image is selected

# YOU SHOULD SET THESE VALUES TO REFLECT THE SETUP OF YOUR CHECKERBOARD
WIDTH = 6
HEIGHT = 9

replaceImg = cv2.imread(images[currentImage])
rows, cols, ch = replaceImg.shape
pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])  # this points are necessary for the transformation

processing = True  # boolean variable using for disabling the image processing
maskThreshold=10

while (True):

    # Capture frame-by-frame
    ret, img = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # This function is used to detect the corners of the chessboard, 9x6 is the number of corners to find
    ret, corners = cv2.findChessboardCorners(gray, (HEIGHT, WIDTH), None)

    # If found, do the processing
    if ret == True and processing:
        # pts2 is used for defining the perspective transform
        pts2 = np.float32([corners[0, 0], corners[8, 0], corners[len(corners) - 1, 0], corners[len(corners) - 9, 0]])
        # compute the transform matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)
        rows, cols, ch = img.shape
        # make the perspective change in a image of the size of the camera input
        dst = cv2.warpPerspective(replaceImg, M, (cols, rows))
        # A mask is created for adding the two images
        # maskThreshold is a variable because that allows to subtract the black background from different images
        ret, mask = cv2.threshold(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), maskThreshold, 1, cv2.THRESH_BINARY_INV)
        # Erode and dilate are used to delete the noise
        mask = cv2.erode(mask, (3, 3))
        mask = cv2.dilate(mask, (3, 3))
        # The two images are added using the mask
        for c in range(0, 3):
            img[:, :, c] = dst[:, :, c] * (1 - mask[:, :]) + img[:, :, c] * mask[:, :]

    # Finally the result is displayed
    cv2.imshow('img', img)

    # Wait for the key
    key = cv2.waitKey(1)

    if key == ord('q'):  # Quitting the window
        print("Quit")
        break

    if key == ord('p'):  # Processing the Image
        processing = not processing
        if processing:
            print("Activated image processing")
        else:
            print("Deactivated image processing")

cap.release()

cv2.destroyAllWindows()