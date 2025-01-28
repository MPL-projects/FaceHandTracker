import cv2
import mediapipe as mp
import time
import numpy as np
from time import time

opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="models/deploy.prototxt",
                                            caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel")


def cvDnnDetectFaces(image, opencv_dnn_model, min_confidence=0.5):
    image_height, image_width, _ = image.shape
    output_image = image.copy()

    preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

    opencv_dnn_model.setInput(preprocessed_image)
    results = opencv_dnn_model.forward()

    for face in results[0][0]:
        face_confidence = face[2]
        if face_confidence > min_confidence:
            bbox = face[3:]
            x1 = int(bbox[0] * image_width)
            y1 = int(bbox[1] * image_height)
            x2 = int(bbox[2] * image_width)
            y2 = int(bbox[3] * image_height)

            cv2.rectangle(output_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=image_width // 200)

    return output_image



def main():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Detection de visages")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Erreur")
            break

        frame = cv2.flip(frame, 1)

        frame = cvDnnDetectFaces(frame, opencv_dnn_model)
        cv2.imshow("Detection de visages", frame)

        # Quitter en appuyant sur ESC
        key = cv2.waitKey(1)
        if key % 256 == 27:
            print("ESC")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
