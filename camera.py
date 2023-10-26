import cv2
# import main as mn
import yolov5
import numpy as np

# # load model
# model = yolov5.load('keremberke/yolov5m-license-plate')

# # set model parameters
# model.conf = 0.25  # NMS confidence threshold
# model.iou = 0.45  # NMS IoU threshold
# model.agnostic = False  # NMS class-agnostic
# model.multi_label = False  # NMS multiple labels per box
# model.max_det = 10  # maximum number of detections per image

# img_test = mn.preprocess_img("dataset\platGray\E536YY.jpg")

# print(mn.predict_image(img_test, mn.prediction_model_loaded))

cam = cv2.VideoCapture(0)
cv2.namedWindow('test')

img_counter = 0

while True:
    ret,frame = cam.read()
    
    if not ret:
        print("failed to grab frame")
        break
    
    cv2.putText(img=frame, 
                text="input text",
                org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(225, 0, 222),
                thickness=2, 
                )
    
    cv2.imshow('test', frame)
    
    k = cv2.waitKey(1)
    
    if k % 256 == 27:
        print("Closing app")
        break
    
    elif k % 256 ==32:
        img_name = "open_Cv_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("ScreenShoot")
        img_counter += 1
        
        # img_test = mn.preprocess_img(img_name)
        # print(mn.predict_image(img_test, mn.prediction_model_loaded))


        # # perform inference
        # results = model(img_name, size=640)

        # # inference with test time augmentation
        # results = model(img_name, augment=True)

        # # parse results
        # predictions = results.pred[0]
        # boxes = predictions[:, :4] # x1, y1, x2, y2
        # # scores = predictions[:, 4]
        # # categories = predictions[:, 5]
        # detect = cv2.rectangle(frame, 
        #                        (0, 0), 
        #                        (100, 100), 
        #                        (255, 0, 222), 
        #                        1
        #                        )
        # cv2.imwrite("detect{}.jpg".format(img_counter), detect)

        # results.save()
        
cam.release()
