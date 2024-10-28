# Python code to read image
#
# From - https://www.geeksforgeeks.org/reading-image-opencv-using-python/
#
# First make sure you have up-to-date PIP and OpenCV on wheels:
#
# curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# # python3 -m pip install --upgrade pip
# python3 -m pip install opencv-python
# python3 -m pip install numpy
# python3 -m pip install matplotlib
#
import cv2

# To read image from disk, we use
# cv2.imread function, in below method,
img = cv2.imread("sun-image.png", cv2.IMREAD_COLOR)

# Creating GUI window to display an image on screen
# first Parameter is windows title (should be in string format)
# Second Parameter is image array
cv2.imshow("image", img)

# To hold the window on screen, we use cv2.waitKey method
# Once it detected the close input, it will release the control
# To the next line
# First Parameter is for holding screen for specified milliseconds
# It should be positive integer. If 0 pass an parameter, then it will
# hold the screen until user close it.
cv2.waitKey(0)

# It is for removing/deleting created GUI window from screen
# and memory
cv2.destroyAllWindows()
