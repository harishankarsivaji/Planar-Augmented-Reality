# Planar Augmented Reality

- To compute the projection from a planar target to a real-world image in real-time.
- Use your web-cam to capture images of a checkerboard which you will track using OpenCV and then overlay another image on top            of the image which is aligned with the checkerboard.
In effect you will use plane-to-plane calibration to perform planar based augmented reality (AR).

http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#warpperspective

Step 1: Planar Homographies
 Calibrate the real camera . 
 (i) compute the distortion (and intrinsic) parameters for your camera.
 (ii) undistort images from your camera such that the lines of the input checkerboard are straightened. 
 
 Step 2: 
 Take an image of the checkerboard and undistort it. Now using the facilities provided by the OpenCV calib3d library generate a set of correspondences between the checkerboard and the undistorted image.
Using these correspondences compute the 3x3 homography (see notes on planar camera model) that maps points on the checkerboard to points on in the image. 

Step 3 : 
Using the wrapprespective function and an image of your choice, project the image such that it appears to lie in the plane of the checkerboard.
