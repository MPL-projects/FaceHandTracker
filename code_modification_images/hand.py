import cv2
import mediapipe as mp
import time
import numpy as np
from time import time

opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="models/deploy.prototxt",
                                            caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Initialize Mediapipe tools
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


def cvDnnDetectFaces(image, opencv_dnn_model, min_confidence=0.5):
    """
    Detect faces in an image using OpenCV DNN model.

    Parameters:
        image (numpy.ndarray): Input image in BGR format.
        opencv_dnn_model: Preloaded OpenCV DNN model.
        min_confidence (float): Minimum confidence to consider a detection valid.

    Returns:
        numpy.ndarray: Annotated image with detected faces.
    """
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

def process_hand_detection(img):
    """
    Process an image for hand detection and annotate key points and connections.

    Parameters:
        img (numpy.ndarray): Input image in BGR format.

    Returns:
        numpy.ndarray: Annotated image with hand landmarks.
    """
    
    # Convert the image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Highlight specific key points
                if id in [4, 8, 12, 16, 20]:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # Draw connections between landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    return img


def main():
    # Real-time detection using webcam
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Hand and Face Detection")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)

        # Process the frame for face detection
        frame = cvDnnDetectFaces(frame, opencv_dnn_model)

        # Process the frame for hand detection
        frame = process_hand_detection(frame)

        # Display the frame
        cv2.imshow("Hand and Face Detection", frame)

        # Exit on pressing ESC
        key = cv2.waitKey(1)
        if key % 256 == 27:
            print("Escape hit, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
