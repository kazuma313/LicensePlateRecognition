import cv2

def camera_forDetection(model_character, model_detection):
    
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('camera')

    img_counter = 0
    fileName_detect_plate = "deteksi_plat.jpg"

    while True:
        ret,frame = cam.read()
        
        if not ret:
            print("failed to grab frame")
            break
        
        img = cv2.imwrite(fileName_detect_plate, frame)
        
        deteksi_plat = model_detection.predict(fileName_detect_plate)

        if deteksi_plat:
            start_point = (deteksi_plat[0]['box']['xmin'], deteksi_plat[0]['box']['ymin'])
            end_point = (deteksi_plat[0]['box']['xmax'], deteksi_plat[0]['box']['ymax'])
            nomor_plat = "".join(model_character.predict_image(fileName_detect_plate))
            
            # put text to image
            cv2.putText(img=frame, 
                        text=nomor_plat,
                        org= start_point,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 200),
                        thickness=2, 
                        )
            
            # put rectangle to image
            cv2.rectangle(frame,
                        start_point, 
                        end_point, 
                        (0, 255, 0), 
                        5
                        )
            
        img = cv2.imwrite("plat_detected.jpg", frame)
        
        # Menampilkan image
        cv2.imshow('camera', frame)
        
        k = cv2.waitKey(1)
        
        if k % 256 == 27:
            print("Closing app")
            break
        
        elif k % 256 ==32:
            img_name = "open_Cv_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("ScreenShoot")
            img_counter += 1

            
    cam.release()


if __name__ == "__main__": 
    import model as mn
    model_character = mn.model_character()
    model_detection = mn.model_platDetection(model_path="skiba4/license_plate")
    camera_forDetection(model_character=model_character, model_detection=model_detection)
    