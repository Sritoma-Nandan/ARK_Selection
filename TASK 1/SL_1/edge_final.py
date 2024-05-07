import numpy as np
import cv2 

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #sobel_x_kernel = np.array([[2,2,4,2,2],[1,1,2,1,1],[0,0,0,0,0],[-1,-1,-2,-1,-1],[-2,-2,-4,-2,-2]])
    #sobel_y_kernel = np.array([[2,1,0,-1,-2],[2,1,0,-1,-2],[4,2,0,-2,-4],[2,1,0,-1,-2],[2,1,0,-1,-2]])
    
    sobel_x_kernel = np.array([[-3, 0, 3],[-10, 0, 10],[-3, 0, 3]])
    sobel_y_kernel = np.array([[-3, -10, -3],[ 0,  0,  0],[ 3,  10,  3]])
    
    sobel_x = cv2.filter2D(gray, -1, sobel_x_kernel)
    sobel_y = cv2.filter2D(gray, -1, sobel_y_kernel)
    #gradient_magnitude = np.sqrt((sobel_x**2) + (sobel_y**2))
    gradient_magnitude= cv2.add(sobel_x,sobel_y)
    
    mask = cv2.inRange(gradient_magnitude,140, 190)
    result = cv2.bitwise_and(gradient_magnitude, gradient_magnitude, mask=mask)
    
    '''gradient_magnitude *= (255.0 / (gradient_magnitude.max()))
    gradient_magnitude = gradient_magnitude.astype(np.uint8)'''
    _, edges = cv2.threshold(result, 128,255, cv2.THRESH_BINARY)
    return edges

def hough_transform(image):
    hough_image = np.copy(image)  
    lines=cv2.HoughLinesP(hough_image,1,np.pi/180,90,minLineLength=10,maxLineGap=45)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line[0]
            if(x2-x1)!=0:
                slope=((y2-y1)/(x2-x1))
                if abs(slope)<1.732:
                    y3=(slope*(0+x2)*(-1))+y2
                    cv2.line(img,(0,int(y3)),(x2,y2),(36,33,187),10)
    else:
        pass
    return img

img=cv2.imread('table.png')
img=cv2.resize(img,(700,700))

edge=sobel_edge_detection(img) 
cv2.imwrite('edge.png',edge)

edge_img=cv2.imread('edge.png',cv2.IMREAD_UNCHANGED)
hough_image = hough_transform(edge_img)
cv2.imshow('Edge Detected', hough_image)

cv2.waitKey(0)
cv2.destroyAllWindows()