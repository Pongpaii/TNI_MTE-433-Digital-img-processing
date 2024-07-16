import cv2
import cvzone
import numpy as np


#declare cap that keep videocapture  0=first argument
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#loop     success = boolean (t/f)
while True:
    success, img = cap.read()

    img_flip = cv2.flip(img,0)
    img_flip2 = cv2.flip(img, 0)
    # imgList = [img_flip, img ,img_flip2]

    b,g,r = cv2.split(img)



    Gaussian_blur=cv2.GaussianBlur(img,(33,33),0)
    Gaussian_blurlarge = cv2.GaussianBlur(img, (99, 99), 0)
    bilateralFilter = cv2.bilateralFilter(img,15,75,75)

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding
    # The threshold value is the mean of the neighbourhood area minus the constant C.
    thresh7 = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 199, 5)
    # The threshold value is a gaussian-weighted sum of the neighbourhood values minus the constant C.
    thresh8 = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 199, 5)

    nbgr = [200, 150, 250]
    # BGColor = np.ones(img.shape, np.uint8)
    BGColor = np.ones_like(img)
    BGColor[:, :, :] = BGColor[:, :, :] * nbgr
    # image background
    BGimg = cv2.imread("resource/gojo1.jpg")
    BGimg2 = cv2.imread("resource/gojo2.jpg")
    BGimg3 = cv2.imread("resource/gojo3.jpg")
    BGimg4 = cv2.imread("resource/skn1.jpg")
    BGimg5 = cv2.imread("resource/skn2.jpg")
    BGimg6 = cv2.imread("resource/skn3.jpg")

    # resize the background image to be the same size as image frame from camera
    BGimg = cv2.resize(BGimg, (img.shape[1], img.shape[0]))
    BGimg2 = cv2.resize(BGimg2, (img.shape[1], img.shape[0]))
    # mask image using bitwise AND



    thresh8_inv = cv2.bitwise_not(thresh8)
    BGimg_mask = cv2.bitwise_and(BGimg, BGimg, mask=thresh8)
    BGimg_mask2 = cv2.bitwise_and(BGimg2, BGimg2, mask=thresh8)
    BGColor_mask = cv2.bitwise_and(BGColor, BGColor, mask=thresh8_inv)
    BGColor_mask = cv2.bitwise_and(BGColor, BGColor, mask=thresh8_inv)



    nimg1 = cv2.add(BGimg_mask2, BGColor_mask)
    nigm2 = cv2.merge([img[:, :, 0], img[:, :, 0]])


    imgList = [img, BGimg_mask,BGColor_mask,nimg1,BGimg3,BGimg4,BGimg5,BGimg6]




    stackedImg = cvzone.stackImages(imgList,cols=4,scale=0.5)
    cv2.imshow("stackedImg",stackedImg)
    #da img stack box


    cv2.waitKey(1)
