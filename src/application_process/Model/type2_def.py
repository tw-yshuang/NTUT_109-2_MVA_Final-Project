import cv2 
import numpy as np
def erode(gray, kernel= None, iterations = 1, isShow=True):
    if kernel is None:
        kernel = np.ones((3,3),np.uint8)
    img_erode = cv2.erode(gray, kernel, iterations = 1)
    # if isShow:
    #     #cv2.imshow("Orignal", img)
    #     cv2.imshow("Erode", img_erode)
    #     cv2.waitKey(0)
    #return img_erode  
    edges = cv2.Canny(gray, 50, 150,apertureSize = 3)
    minLineLength = 200
    maxLineGap = 5
    for i in range( 60 ,65 , 5):
        threshold = i
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength, maxLineGap)
        print("Find no of line:", len(lines))
    #print(lines[0])
        img_cpy = img_erode.copy()
    ## draw all lines
        for h in lines:
            (x1,y1,x2,y2) = h[0]
            img_ht = cv2.line(img_cpy,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow("Hough line: " + str(threshold), img_ht)
        cv2.waitKey(0)
    return img_erode

def main():
    img = cv2.imread(r'data\train_images\2\00ac8372f.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
    img_erode = erode(gray, kernel, 1)
    #img_hough = hough()
    # cv2.waitKey(0)
    return


if __name__ == "__main__":
    main()
