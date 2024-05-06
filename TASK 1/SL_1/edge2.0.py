import numpy as np
import cv2 

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sobel_x_kernel = np.array([[2,2,4,2,2],[1,1,2,1,1],[0,0,0,0,0],[-1,-1,-2,-1,-1],[-2,-2,-4,-2,-2]])
    sobel_y_kernel = np.array([[2,1,0,-1,-2],[2,1,0,-1,-2],[4,2,0,-2,-4],[2,1,0,-1,-2],[2,1,0,-1,-2]])
    
    '''sobel_x_kernel = np.array([[-3, 0, 3],[-10, 0, 10],[-3, 0, 3]])
    sobel_y_kernel = np.array([[-3, -10, -3],[ 0,  0,  0],[ 3,  10,  3]])'''
    
    sobel_x = cv2.filter2D(gray, -1, sobel_x_kernel)
    sobel_y = cv2.filter2D(gray, -1, sobel_y_kernel)
    
    
    gradient_magnitude = np.sqrt((sobel_x**2) + (sobel_y**2))
    gradient_magnitude *= (255.0 / (gradient_magnitude.max()))
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    edges = cv2.threshold(gradient_magnitude, 250, 255, cv2.THRESH_BINARY)[1]
    
    return edges

def hough_transform(image):
    hough_image = np.copy(image)  
    lines=cv2.HoughLinesP(hough_image,1,np.pi/180,90,minLineLength=90,maxLineGap=10)
    if lines is not None:
        for line in lines:
            #print(line)
            x1,y1,x2,y2=line[0]
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),5)
    else:
        pass
    return img

img=cv2.imread('table.png')
img=cv2.resize(img,(700,700))

edge=sobel_edge_detection(img) 
cv2.imshow('Edge Image', edge)

hough_image = hough_transform(edge)
cv2.imshow('Edge Detected', hough_image)

cv2.waitKey(0)
cv2.destroyAllWindows()