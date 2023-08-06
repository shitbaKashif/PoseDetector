import cv2
import numpy as np
from urllib import request
import matplotlib.pyplot as plt

# Load the pre-trained face, eye, and smile cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# Check if the cascade classifiers loaded successfully
if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    print('Error: Cascade classifiers not loaded.')
else:
    # Get the image from the web (replace the URL with the image URL you want to use)
    response = request.urlopen('https://as2.ftcdn.net/v2/jpg/02/46/14/95/1000_F_246149573_1dbnEopMZjSflWG4ZvojXhVVV8cTewTW.jpg')
    data = response.read()
    # data = cv2.imread('./S2.jpg')
    img_data = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Function to calculate the eye aspect ratio (EAR)
    def eye_aspect_ratio(eye):
        max = len(eye)
        # Euclidean distance between the vertical eye landmarks (2 and 6)
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[max-4]))
        # Euclidean distance between the horizontal eye landmarks (1 and 4)
        B = np.linalg.norm(np.array(eye[3]) - np.array(eye[max-2]))
        # Euclidean distance between the diagonal eye landmarks (0 and 3)
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[max-1]))
        ear = (A + B) / (2.0 * C)
        return ear

    # Function to calculate the mouth aspect ratio (MAR)
    def mouth_aspect_ratio(mouth):
        max = len(mouth)
        A = np.linalg.norm(mouth[1] - mouth[max-1])  # Distance between the corners of the mouth (vertical)
        B = np.linalg.norm(mouth[2] - mouth[max-2])  # Distance between the corners of the mouth (vertical)
        C = np.linalg.norm(mouth[0] - mouth[max-3])  # Width of the mouth

        mar = (A + B) / (2.0 * C)
        return mar

    # Function to check if the mouth is open
    def is_mouth_open(mouth):
        # MAR threshold for mouth detection (adjust this threshold as needed)
        threshold = 0.75
        mar = mouth_aspect_ratio(mouth)
        return mar > threshold

    # Function to check if the user is facing the camera
    def is_facing_camera(eyes):
        if len(eyes) == 2:
            eye1, eye2 = eyes
            # Calculate the distance between the centers of the eyes
            distance = np.linalg.norm(eye1[0] - eye2[0])
            # Set a threshold for symmetric eye positions
            threshold_distance = 0.2 * distance  # You can adjust this threshold based on your use case
            # Calculate the eye aspect ratio for both eyes
            ear1 = eye_aspect_ratio(eye1)
            ear2 = eye_aspect_ratio(eye2)
            # Check if both eyes are looking towards the center of the image based on EAR values
            return abs(ear1 - ear2) < threshold_distance
        else:
            return False

    # Iterate over detected faces, find eyes and smiles, and draw rectangles around found objects
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Check if the user is facing the camera based on eye positions
        facing_camera = is_facing_camera(eyes)

        # Calculate eye aspect ratio for each detected eye
        ear_values = []
        for (ex, ey, ew, eh) in eyes:
            eye = [(ex, ey), (ex + ew // 2, ey + eh // 2), (ex + ew, ey), (ex + ew // 2, ey + eh), (ex, ey + eh // 2), (ex + ew, ey + eh // 2)]
            ear = eye_aspect_ratio(eye)
            if ear is not None:  # Check if the eye aspect ratio is calculated successfully
                ear_values.append(ear)
                threshold_ear = 0.2
                if ear > threshold_ear:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)           

        smiles = smile_cascade.detectMultiScale(roi_gray)
        mouths = mouth_cascade.detectMultiScale(roi_gray)
        # Calculate smile detection score for each detected smile
        smile_values = []
        for (sx, sy, sw, sh) in smiles:
            if sw > 100 :
                smile_roi = roi_gray[sy:sy + sh, sx:sx + sw]
                smile_score = mouth_aspect_ratio(smile_roi)
                if smile_score > 1:  
                    smile_values.append(smile_score)
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
                    print("SSS",smile_score)
        # mouth_values = []
        # for (mx, my, mw, mh) in mouths:
        #     if sw > 100 :
        #         mouth_roi = roi_gray[my:my + mh, mx:mx + mw]
        #         mouth_score = mouth_aspect_ratio(mouth_roi)
        #         if mouth_score is not None:  # Check if the smile detection score is calculated successfully
        #             mouth_values.append(mouth_score)
        #             if mouth_score > 0.2:
        #                 cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)
        # print(is_mouth_open(mouths[0]))
        # Update the scores
        is_facing_camera_score = 1.0 if facing_camera else 0.0
        eyes_directed_camera_score = sum(ear_values) / len(ear_values) if ear_values else 0.0
        # mouth_open_score = sum(mouth_values) / len(mouth_values) if mouth_values else 0.0
        mouth_open_score = sum(1 for mouth in mouths if is_mouth_open(mouth)) / len(mouths) if len(mouths) > 0 else 0.0
        is_smiling_score = sum(smile_values) / len(smile_values) if smile_values else 0.0

        # Display the scores on the terminal
        print("Facing Camera:", is_facing_camera_score)
        print("Eyes Directed:", eyes_directed_camera_score)
        print("Mouth Open:", mouth_open_score)
        print("Is Smiling:", is_smiling_score)

    # Show the image with detections
    plt.imshow(img)
    plt.show()
