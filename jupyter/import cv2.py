import cv2
for i in range(256,275):
    img=cv2.imread("D:/Bilder/zwiebel/zwiebel("+str(i)+").jpg")
    cv2.imwrite("D:/Bilder/zwiebel/zwiebel_"+str(i)+".jpg",img)