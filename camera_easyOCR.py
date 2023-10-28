import cv2
import easyocr
reader = easyocr.Reader(['en'])

cam = cv2.VideoCapture(0)
cv2.namedWindow('test')

img_counter = 0

while True:
    ret,frame = cam.read()
    
    if not ret:
        print("failed to grab frame")
        break
    
    result = reader.readtext(frame)

        # put text to image
    if result:
        cv2.putText(img=frame, 
                    text= " ".join([i[-2] for i in result[:3]]),
                    org= (50, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 200),
                    thickness=2, 
                    )
        
    
    # Menampilkan image
    cv2.imshow('test', frame)
    
    k = cv2.waitKey(1)
    
    if k % 256 == 27:
        print("Closing app")
        break
    
    elif k % 256 ==32:
        img_name = "open_Cv_easyOCR{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("ScreenShoot")
        img_counter += 1

        
cam.release()