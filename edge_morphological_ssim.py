import cv2
import cvzone
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
 success, img = cap.read()
 imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

 # Apply Canny edge detection on the grayscale image
 imgCanny = cv2.Canny(imgGray, 50, 150)
# cv2.imshow("ImgCanny", imgCanny)

 # Morphological Operations
 # define the kernel (structuring element)
 kernel_L = np.ones((9, 9), np.uint8)
 kernel_M = np.ones((7, 7), np.uint8)
 kernel_S = np.ones((3, 3), np.uint8)
 # dilate the image (white region increases)
 dilation = cv2.dilate(imgCanny, kernel_M, iterations=1)
 # cv2.imshow("ImgCanny_Dilate", dilation)
 # erode the image (white region decreases)
 erosion = cv2.erode(dilation, kernel_M, iterations=1)
 # cv2.imshow("ImgCanny_Erosion", erosion)


 # closing the image (dilation followed by Erosion to close small black points on the white object)
 closing = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, kernel_M, iterations=1)
 # opening the image (erosion followed by dilation to remove noise)
 opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_S, iterations=1)
 # morphological gradient (dilation - erosion)
 morph_gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel_S)

 # Structural Similarity (SSIM) Index
 Gaussian_base = cv2.GaussianBlur(imgGray, (5, 5), 0)
 Gaussian_blur = cv2.GaussianBlur(imgGray, (55, 55), 0)
 # compute the Structural Similarity (SSIM) Index between the two images
 (score, diff) = compare_ssim(Gaussian_base, Gaussian_blur, full=True)
 diff = (diff * 255).astype("uint8")

 Gaussian_blur255 = cv2.GaussianBlur(imgGray, (255, 255), 0)
 (score, diff2) = compare_ssim(Gaussian_base, Gaussian_blur255, full=True)
 diff_max = (diff2 * 255).astype("uint8")




 imgList = [img, imgGray, imgCanny, dilation, erosion,
            closing, opening, morph_gradient, Gaussian_base, Gaussian_blur, diff,Gaussian_blur255 ,diff]
 #imgList = [img, imgGray, imgCanny, dilation, erosion,closing,opening,morph_gradient]
 # Stack the images together using cvzone's stackImages function
 stackedImg = cvzone.stackImages(imgList, cols=5, scale=0.5)


 # Display the stacked images
 cv2.imshow("stackedImg", stackedImg)


 key = cv2.waitKey(1)
 # if cv2.waitKey(1) & 0xff == 27: # ESC
 if key == ord('q'):
   break