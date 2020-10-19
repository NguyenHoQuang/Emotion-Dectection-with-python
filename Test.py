import datetime
from datetime import  datetime

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import tensorflow as tf

face_classifier = cv2.CascadeClassifier('/D:/Data/UIT_Project/Emotion Detect with Python/haarcascade_frontalface_default.xml')
classifier = load_model('/D:/Data/UIT_Project/Emotion Detect with Python/Emotion_little_vgg.h5')

class_labels = ['angry', 'happy', 'neutral', 'sad', 'surprise']



cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    labels = []
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        # rect,face,image = face_detector(frame)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class
            #Đôi khi, bạn sẽ phải chơi với các vùng hình ảnh nhất định.
            # Để phát hiện mắt trong hình ảnh, phát hiện khuôn mặt đầu tiên được thực hiện trên toàn bộ hình ảnh.
            # Khi có được một khuôn mặt, chúng ta chọn vùng mặt một mình và tìm kiếm mắt bên trong nó thay vì tìm kiếm toàn bộ hình ảnh.
            # Nó cải thiện độ chính xác (vì mắt luôn ở trên khuôn mặt) ​​và hiệu suất (vì chúng ta tìm kiếm trong một khu vực nhỏ).

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



