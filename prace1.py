import cv2
import cvzone
import numpy as np


#declare cap that keep videocapture  0=first argument
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#loop     success = boolean (t/f)
while True:
    success, img = cap.read()

    img_flip = cv2.flip(img,1)
    img_flip2 = cv2.flip(img, 0)
    # imgList = [img_flip, img ,img_flip2]

    b,g,r = cv2.split(img)



    Gaussian_blur=cv2.GaussianBlur(img,(33,33),0)
    Gaussian_blurlarge = cv2.GaussianBlur(img, (99, 99), 0)
    bilateralFilter = cv2.bilateralFilter(img,15,75,75)

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)
    cvzone.putTextRect(thresh1, "Binary", (20, 50), colorR=(0, 0, 0), offset=20)
    ret, thresh2 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TOZERO_INV)
    # Otsu's algorithm finds the best threshold value to minimize the weighted within-class variance.
    ret, thresh6 = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
    BGimg = cv2.imread("Resource/maxresdefault.webp")
    # resize the background image to be the same size as image frame from camera
    BGimg = cv2.resize(BGimg, (img.shape[1], img.shape[0]))
    # mask image using bitwise AND
    BGimg_mask = cv2.bitwise_and(BGimg, BGimg, mask=thresh8)
    thresh8_inv = cv2.bitwise_not(thresh8)
    BGColor_mask = cv2.bitwise_and(BGColor, BGColor, mask=thresh8_inv)
    nimg1 = cv2.add(BGimg_mask, BGColor_mask)
    nigm2 = cv2.merge([img[:, :, 0], img[:, :, 0]])
    imgList = [img, BGimg_mask,BGColor_mask]

    imgGrayflip = cv2.cvtColor(img_flip, cv2.COLOR_BGR2GRAY)
    Gaussian_blur2 = cv2.GaussianBlur(img_flip, (55, 55), 0)

    imgList = [img_flip, imgGrayflip,Gaussian_blur2]

    stackedImg = cvzone.stackImages(imgList,cols=4,scale=0.5)
    cv2.imshow("stackedImg",stackedImg)
    #da img stack box


    cv2.waitKey(1)
