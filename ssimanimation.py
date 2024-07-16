import cv2
import cvzone
from skimage.metrics import structural_similarity as compare_ssim
import imageio
img = cv2.imread("resource/meme.jpg") # Cartoon
img = cv2.resize(img, (640, 480))
# Convert the image to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# SSIM Animation
Gaussian_base = cv2.GaussianBlur(imgGray, (5, 5), 0)
SSIMList = []


for i in range(11, 400, 10):
 print(i)
 # Gaussian Blurring
 Gaussian_blur = cv2.GaussianBlur(imgGray, (i, i), 0)
 # compute the Structural Similarity Index (SSIM) between the two images
 (score, diff) = compare_ssim(Gaussian_base, Gaussian_blur, full=True)
 diff = (diff * 255).astype("uint8")
 SSIMList.append(diff)
# Stack the images together using cvzone's stackImages function
stackedImg = cvzone.stackImages(SSIMList, cols=10, scale=0.25)
# Display the stacked images
cv2.imshow("stackedImg", stackedImg)
cv2.waitKey(0)

with imageio.get_writer("SSIM.gif", mode="I") as writer:
  for idx, frame in enumerate(SSIMList):
    print("Adding frame to GIF file: ", idx + 1)
    writer.append_data(frame)