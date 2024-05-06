import cv2
import numpy as np

def generate_depth_map(left_img, right_img):
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    #stereo = cv2.StereoSGBM_create(minDisparity=16,numDisparities=16, blockSize=15,P1=8*3*15*15,P2=32*3*15*15,disp12MaxDiff=1,uniquenessRatio=5,speckleWindowSize=0,speckleRange=2,preFilterCap=63,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)
    disparity = stereo.compute(gray_left, gray_right)
    #diff = np.abs(gray_left.astype(np.float32), gray_right.astype(np.float32))
    disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disp_normalized


left_image = cv2.imread('left.png')
right_image = cv2.imread('right.png')
depth_map = generate_depth_map(left_image, right_image)
cv2.imshow('hh',depth_map)
colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
cv2.imshow('Depth Map Colormap', colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()
