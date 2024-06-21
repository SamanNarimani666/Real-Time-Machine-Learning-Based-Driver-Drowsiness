from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import math
import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
import mediapipe as mp
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# Initialize the MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize the video stream and sleep for a bit, allowing the camera sensor to warm up
print("[INFO] initializing camera...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 400x225 to 1024x576
frame_width = 1920
frame_height = 1080

# 2D image points. If you change the image, you need to change vector
image_points = np.array([
    (359, 391),     # Nose tip 34
    (399, 561),     # Chin 9
    (337, 297),     # Left eye left corner 37
    (513, 301),     # Right eye right corner 46
    (345, 465),     # Left Mouth corner 49
    (453, 469)      # Right mouth corner 55
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0

data = []
labels = []

while True:
    # Grab the frame from the threaded video stream, resize it to have a maximum width of 400 pixels, and convert it to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=1024, height=576)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Convert back to color to draw colored rectangles and text
    frame_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Check to see if a face was detected, and if so, draw the total number of faces on the frame
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame_gray, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Loop over the face detections
    for rect in rects:
        # Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Convert the dlib rectangle to OpenCV bounding box [i.e., (x, y, w, h)] and draw the face bounding box
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

        # Loop over the (x, y)-coordinates for the facial landmarks and draw each of them
        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # Average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Compute the convex hull for the left and right eye, then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame_gray, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame_gray, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.putText(frame_gray, "EAR: {:.2f}".format(ear), (480, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            # If the eyes were closed for a sufficient number of times, then show the warning
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame_gray, "Eyes Closed!", (520, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)                
        else:
            COUNTER = 0

        # Compute the mouth aspect ratio
        mouth = shape[mStart:mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame_gray, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame_gray, "MAR: {:.2f}".format(mar), (700, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw text if mouth is open
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame_gray, "Yawning!", (900, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Loop over the (x, y)-coordinates for the facial landmarks and draw each of them
        for (i, (x, y)) in enumerate(shape):
            if i == 33:
                image_points[0] = np.array([x, y], dtype='double')
                cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 8:
                image_points[1] = np.array([x, y], dtype='double')
                cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 36:
                image_points[2] = np.array([x, y], dtype='double')
                cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 45:
                image_points[3] = np.array([x, y], dtype='double')
                cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 48:
                image_points[4] = np.array([x, y], dtype='double')
                cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255

, 0), 1)
            elif i == 54:
                image_points[5] = np.array([x, y], dtype='double')
                cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                cv2.circle(frame_gray, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Draw the determinant image points onto the person's face
        for p in image_points:
            cv2.circle(frame_gray, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

        cv2.line(frame_gray, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame_gray, start_point, end_point_alt, (0, 0, 255), 2)

        if head_tilt_degree:
            cv2.putText(frame_gray, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose_2d = (int(face_landmarks.landmark[1].x * frame_width), int(face_landmarks.landmark[1].y * frame_height))
                nose_3d = (int(face_landmarks.landmark[1].x * frame_width), int(face_landmarks.landmark[1].y * frame_height), face_landmarks.landmark[1].z)
                cv2.circle(frame_gray, nose_2d, 5, (0, 255, 0), -1)

                x_center = nose_3d[0] / frame_width
                y_center = nose_3d[1] / frame_height
                z_center = nose_3d[2]

                x_angle = (x_center - 0.5) * 90
                y_angle = (y_center - 0.5) * -90

                if x_angle < -7:
                    cv2.putText(frame_gray, "Head: Left", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif x_angle > 7:
                    cv2.putText(frame_gray, "Head: Right", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if y_angle < -5:
                    cv2.putText(frame_gray, "Head: Down", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif y_angle > 5:
                    cv2.putText(frame_gray, "Head: Up", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the data and labels
        data.append([ear, mar, head_tilt_degree[0] if head_tilt_degree else 0])
        labels.append(1 if mar > MOUTH_AR_THRESH or ear < EYE_AR_THRESH else 0)

    # Show the grayscale frame with colored annotations
    cv2.imshow("Frame", frame_gray)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


print("[INFO] Balancing data...")
sm = SMOTE(random_state=42)
data, labels = sm.fit_resample(data, labels)

print("[INFO] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

print("[INFO] Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)
rf.fit(X_train, y_train)

print("[INFO] Training Neural Network model...")
nn = MLPClassifier(hidden_layer_sizes=(5, 5, 5, 5), max_iter=1000, random_state=42)
nn.fit(X_train, y_train)

print("[INFO] Training SVM model...")
svm = SVC(kernel='linear', C=1, probability=True, random_state=42)
svm.fit(X_train, y_train)

print("[INFO] Evaluating models...")
models = {'Random Forest': rf, 'Neural Network': nn, 'SVM': svm}
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"{name} Model")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# from scipy.spatial import distance as dist
# from imutils.video import VideoStream
# from imutils import face_utils
# import argparse
# import imutils
# import time
# import dlib
# import math
# import cv2
# import numpy as np
# from EAR import eye_aspect_ratio
# from MAR import mouth_aspect_ratio
# from HeadPose import getHeadTiltAndCoords
# import mediapipe as mp

# # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
# print("[INFO] loading facial landmark predictor...")
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# # Initialize the MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# # Initialize the video stream and sleep for a bit, allowing the camera sensor to warm up
# print("[INFO] initializing camera...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

# # 400x225 to 1024x576
# frame_width = 1920
# frame_height = 1080

# # 2D image points. If you change the image, you need to change vector
# image_points = np.array([
#     (359, 391),     # Nose tip 34
#     (399, 561),     # Chin 9
#     (337, 297),     # Left eye left corner 37
#     (513, 301),     # Right eye right corner 46
#     (345, 465),     # Left Mouth corner 49
#     (453, 469)      # Right mouth corner 55
# ], dtype="double")

# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# (mStart, mEnd) = (49, 68)

# EYE_AR_THRESH = 0.25
# MOUTH_AR_THRESH = 0.79
# EYE_AR_CONSEC_FRAMES = 3
# COUNTER = 0

# while True:
#     # Grab the frame from the threaded video stream, resize it to have a maximum width of 400 pixels, and convert it to grayscale
#     frame = vs.read()
#     frame = imutils.resize(frame, width=1024, height=576)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     size = gray.shape

#     # Detect faces in the grayscale frame
#     rects = detector(gray, 0)

#     # Convert back to color to draw colored rectangles and text
#     frame_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

#     # Check to see if a face was detected, and if so, draw the total number of faces on the frame
#     if len(rects) > 0:
#         text = "{} face(s) found".format(len(rects))
#         cv2.putText(frame_gray, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     # Loop over the face detections
#     for rect in rects:
#         # Compute the bounding box of the face and draw it on the frame
#         (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
#         cv2.rectangle(frame_gray, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
#         # Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)

#         # Extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)
#         # Average the eye aspect ratio together for both eyes
#         ear = (leftEAR + rightEAR) / 2.0

#         # Compute the convex hull for the left and right eye, then visualize each of the eyes
#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame_gray, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame_gray, [rightEyeHull], -1, (0, 255, 0), 1)
#         cv2.putText(frame_gray, "EAR: {:.2f}".format(ear), (480, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # Check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
#         if ear < EYE_AR_THRESH:
#             COUNTER += 1
#             # If the eyes were closed for a sufficient number of times, then show the warning
#             if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                 cv2.putText(frame_gray, "Eyes Closed!", (520, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)                
#         else:
#             COUNTER = 0

#         # Compute the mouth aspect ratio
#         mouth = shape[mStart:mEnd]
#         mouthMAR = mouth_aspect_ratio(mouth)
#         mar = mouthMAR
#         mouthHull = cv2.convexHull(mouth)
#         cv2.drawContours(frame_gray, [mouthHull], -1, (0, 255, 0), 1)
#         cv2.putText(frame_gray, "MAR: {:.2f}".format(mar), (700, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # توضیحات مرتبط با انتخاب مقدار آستانه‌ی MAR
#         # اکثریت رانندگان هنگام خمیازه کشیدن به مقدار حداکثر MAR (MAX MAR) برابر با 0.9 می‌رسند.
#         # رانندگانی که دهان کوچکتری دارند، ممکن است به مقادیر حداکثر 0.6 یا 0.7 برسند، بسته به اندازه دهان و نحوه خمیازه کشیدن.
#         # برای انتخاب بهترین مقدار آستانه‌ی MAR، چندین آزمایش انجام شد.
#         # در صورتی که مقدار آستانه بیشتر از 0.9 باشد، تمام فریم‌های داده‌ی تمامی افراد با برچسب "دهان بسته" علامت‌گذاری می‌شوند، بدون توجه به وضعیت واقعی دهان.
#         # برای مقادیر آستانه‌ی بین 0.8 و 0.6 نیز مشکلات مشابهی مشاهده شد.
#         # در مقدار آستانه‌ی 0.5، توانستند همه افراد را به درستی با برچسب "دهان باز" یا "دهان بسته" علامت‌گذاری کنند.
#         # هر مقدار آستانه‌ای کمتر از 0.5 باعث برچسب‌گذاری نادرست برخی فریم‌ها در مواقعی مانند صحبت کردن یا خندیدن می‌شد.
#         # بنابراین، مقدار آستانه‌ی MAR به حداقل 0.5 تنظیم شد.

#         # Draw text if mouth is open
#         if mar > MOUTH_AR_THRESH:
#             cv2.putText(frame_gray, "Yawning!", (900, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # Loop over the (x, y)-coordinates for the facial landmarks and draw each of them
#         for (i, (x, y)) in enumerate(shape):
#             if i == 33:
#                 image_points[0] = np.array([x, y], dtype='double')
#                 cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 8:
#                 image_points[1] = np.array([x, y], dtype='double')
#                 cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 36:
#                 image_points[2] = np.array([x, y], dtype='double')
#                 cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 45:
#                 image_points[3] = np.array([x, y], dtype='double')
#                 cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 48:
#                 image_points[4] = np.array([x, y], dtype='double')
#                 cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 54:
#                 image_points[5] = np.array([x, y], dtype='double')
#                 cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             else:
#                 cv2.circle(frame_gray, (x, y), 1, (0, 0, 255), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

#         # Draw the determinant image points onto the person's face
#         for p in image_points:
#             cv2.circle(frame_gray, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

#         (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

#         cv2.line(frame_gray, start_point, end_point, (255, 0, 0), 2)
#         cv2.line(frame_gray, start_point, end_point_alt, (0, 0, 255), 2)

#         if head_tilt_degree:
#             cv2.putText(frame_gray, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
#         # توضیحات مربوط به محاسبه زاویه چرخش سر
#         results = face_mesh.process(frame)

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 nose_2d = (int(face_landmarks.landmark[1].x * frame_width), int(face_landmarks.landmark[1].y * frame_height))
#                 nose_3d = (int(face_landmarks.landmark[1].x * frame_width), int(face_landmarks.landmark[1].y * frame_height), face_landmarks.landmark[1].z)
#                 cv2.circle(frame_gray, nose_2d, 5, (0, 255, 0), -1)

#                 x_center = nose_3d[0] / frame_width
#                 y_center = nose_3d[1] / frame_height
#                 z_center = nose_3d[2]

#                 x_angle = (x_center - 0.5) * 90
#                 y_angle = (y_center - 0.5) * -90

#                 if x_angle < -7:
#                     cv2.putText(frame_gray, "Head: Left", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                 elif x_angle > 7:
#                     cv2.putText(frame_gray, "Head: Right", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
#                 if y_angle < -5:
#                     cv2.putText(frame_gray, "Head: Down", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                 elif y_angle > 5:
#                     cv2.putText(frame_gray, "Head: Up", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     # Show the grayscale frame with colored annotations
#     cv2.imshow("Frame", frame_gray)
#     key = cv2.waitKey(1) & 0xFF

#     # If the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break

# # Do a bit of cleanup
# cv2.destroyAllWindows()
# vs.stop()


# from scipy.spatial import distance as dist
# from imutils.video import VideoStream
# from imutils import face_utils
# import argparse
# import imutils
# import time
# import dlib
# import math
# import cv2
# import numpy as np
# from EAR import eye_aspect_ratio
# from MAR import mouth_aspect_ratio
# from HeadPose import getHeadTiltAndCoords
# import mediapipe as mp

# # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
# print("[INFO] loading facial landmark predictor...")
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# # Initialize the MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# # Initialize the video stream and sleep for a bit, allowing the camera sensor to warm up
# print("[INFO] initializing camera...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

# # 400x225 to 1024x576
# frame_width = 1920
# frame_height = 1080

# # 2D image points. If you change the image, you need to change vector
# image_points = np.array([
#     (359, 391),     # Nose tip 34
#     (399, 561),     # Chin 9
#     (337, 297),     # Left eye left corner 37
#     (513, 301),     # Right eye right corner 46
#     (345, 465),     # Left Mouth corner 49
#     (453, 469)      # Right mouth corner 55
# ], dtype="double")

# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# (mStart, mEnd) = (49, 68)

# EYE_AR_THRESH = 0.25
# MOUTH_AR_THRESH = 0.79
# EYE_AR_CONSEC_FRAMES = 3
# COUNTER = 0

# while True:
#     # Grab the frame from the threaded video stream, resize it to have a maximum width of 400 pixels, and convert it to grayscale
#     frame = vs.read()
#     frame = imutils.resize(frame, width=1024, height=576)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     size = gray.shape

#     # Detect faces in the grayscale frame
#     rects = detector(gray, 0)

#     # Convert back to color to draw colored rectangles and text
#     frame_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

#     # Check to see if a face was detected, and if so, draw the total number of faces on the frame
#     if len(rects) > 0:
#         text = "{} face(s) found".format(len(rects))
#         cv2.putText(frame_gray, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     # Loop over the face detections
#     for rect in rects:
#         # Compute the bounding box of the face and draw it on the frame
#         (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
#         cv2.rectangle(frame_gray, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
#         # Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)

#         # Extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)
#         # Average the eye aspect ratio together for both eyes
#         ear = (leftEAR + rightEAR) / 2.0

#         # Compute the convex hull for the left and right eye, then visualize each of the eyes
#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame_gray, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame_gray, [rightEyeHull], -1, (0, 255, 0), 1)
#         cv2.putText(frame_gray, "EAR: {:.2f}".format(ear), (480, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # Check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
#         if ear < EYE_AR_THRESH:
#             COUNTER += 1
#             # If the eyes were closed for a sufficient number of times, then show the warning
#             if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                 cv2.putText(frame_gray, "Eyes Closed!", (520, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)                
#         else:
#             COUNTER = 0

#         # Compute the mouth aspect ratio
#         mouth = shape[mStart:mEnd]
#         mouthMAR = mouth_aspect_ratio(mouth)
#         mar = mouthMAR
#         mouthHull = cv2.convexHull(mouth)
#         cv2.drawContours(frame_gray, [mouthHull], -1, (0, 255, 0), 1)
#         cv2.putText(frame_gray, "MAR: {:.2f}".format(mar), (700, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # Draw text if mouth is open
#         if mar > MOUTH_AR_THRESH:
#             cv2.putText(frame_gray, "Yawning!", (900, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # Loop over the (x, y)-coordinates for the facial landmarks and draw each of them
#         for (i, (x, y)) in enumerate(shape):
#             if i == 33:
#                 image_points[0] = np.array([x, y], dtype='double')
#                 cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 8:
#                 image_points[1] = np.array([x, y], dtype='double')
#                 cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 36:
#                 image_points[2] = np.array([x, y], dtype='double')
#                 cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 45:
#                 image_points[3] = np.array([x, y], dtype='double')
#                 cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 48:
#                 image_points[4] = np.array([x, y], dtype='double')
#                 cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 54:
#                 image_points[5] = np.array([x, y], dtype='double')
#                 cv2.circle(frame_gray, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             else:
#                 cv2.circle(frame_gray, (x, y), 1, (0, 0, 255), -1)
#                 cv2.putText(frame_gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

#         # Draw the determinant image points onto the person's face
#         for p in image_points:
#             cv2.circle(frame_gray, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

#         (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

#         cv2.line(frame_gray, start_point, end_point, (255, 0, 0), 2)
#         cv2.line(frame_gray, start_point, end_point_alt, (0, 0, 255), 2)

#         if head_tilt_degree:
#             cv2.putText(frame_gray, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
#         # توضیحات مربوط به محاسبه زاویه چرخش سر
#         results = face_mesh.process(frame)

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 nose_2d = (int(face_landmarks.landmark[1].x * frame_width), int(face_landmarks.landmark[1].y * frame_height))
#                 nose_3d = (int(face_landmarks.landmark[1].x * frame_width), int(face_landmarks.landmark[1].y * frame_height), face_landmarks.landmark[1].z)
#                 cv2.circle(frame_gray, nose_2d, 5, (0, 255, 0), -1)

#                 x_center = nose_3d[0] / frame_width
#                 y_center = nose_3d[1] / frame_height
#                 z_center = nose_3d[2]

#                 x_angle = (x_center - 0.5) * 90
#                 y_angle = (y_center - 0.5) * -90

#                 if x_angle < -7:
#                     cv2.putText(frame_gray, "Head: Left", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                 elif x_angle > 7:
#                     cv2.putText(frame_gray, "Head: Right", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
#                 if y_angle < -5:
#                     cv2.putText(frame_gray, "Head: Down", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                 elif y_angle > 5:
#                     cv2.putText(frame_gray, "Head: Up", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     # Show the grayscale frame with colored annotations
#     cv2.imshow("Frame", frame_gray)
#     key = cv2.waitKey(1) & 0xFF

#     # If the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break

# # Do a bit of cleanup
# cv2.destroyAllWindows()
# vs.stop()

# #!/usr/bin/env python
# from scipy.spatial import distance as dist
# from imutils.video import VideoStream
# from imutils import face_utils
# import argparse
# import imutils
# import time
# import dlib
# import math
# import cv2
# import numpy as np
# from EAR import eye_aspect_ratio
# from MAR import mouth_aspect_ratio
# from HeadPose import getHeadTiltAndCoords
# import mediapipe as mp

# # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
# print("[INFO] loading facial landmark predictor...")
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# # Initialize the MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# # Initialize the video stream and sleep for a bit, allowing the camera sensor to warm up
# print("[INFO] initializing camera...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

# # 400x225 to 1024x576
# frame_width = 1920
# frame_height = 1080

# # 2D image points. If you change the image, you need to change vector
# image_points = np.array([
#     (359, 391),     # Nose tip 34
#     (399, 561),     # Chin 9
#     (337, 297),     # Left eye left corner 37
#     (513, 301),     # Right eye right corne 46
#     (345, 465),     # Left Mouth corner 49
#     (453, 469)      # Right mouth corner 55
# ], dtype="double")

# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# (mStart, mEnd) = (49, 68)

# EYE_AR_THRESH = 0.25
# MOUTH_AR_THRESH = 0.79
# EYE_AR_CONSEC_FRAMES = 3
# COUNTER = 0

# while True:
#     # Grab the frame from the threaded video stream, resize it to have a maximum width of 400 pixels, and convert it to grayscale
#     frame = vs.read()
#     frame = imutils.resize(frame, width=1024, height=576)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     size = gray.shape

#     # Detect faces in the grayscale frame
#     rects = detector(gray, 0)

#     # Check to see if a face was detected, and if so, draw the total number of faces on the frame
#     if len(rects) > 0:
#         text = "{} face(s) found".format(len(rects))
#         cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     # Loop over the face detections
#     for rect in rects:
#         # Compute the bounding box of the face and draw it on the frame
#         (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
#         cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
#         # Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)

#         # Extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)
#         # Average the eye aspect ratio together for both eyes
#         ear = (leftEAR + rightEAR) / 2.0

#         # Compute the convex hull for the left and right eye, then visualize each of the eyes
#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

#         # Check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
#         if ear < EYE_AR_THRESH:
#             COUNTER += 1
#             # If the eyes were closed for a sufficient number of times, then show the warning
#             if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                 cv2.putText(frame, "Eyes Closed!", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         else:
#             COUNTER = 0

#         # Compute the mouth aspect ratio
#         mouth = shape[mStart:mEnd]
#         mouthMAR = mouth_aspect_ratio(mouth)
#         mar = mouthMAR
#         mouthHull = cv2.convexHull(mouth)
#         cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
#         cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # Draw text if mouth is open
#         if mar > MOUTH_AR_THRESH:
#             cv2.putText(frame, "Yawning!", (800, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # Loop over the (x, y)-coordinates for the facial landmarks and draw each of them
#         for (i, (x, y)) in enumerate(shape):
#             if i == 33:
#                 image_points[0] = np.array([x, y], dtype='double')
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 8:
#                 image_points[1] = np.array([x, y], dtype='double')
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 36:
#                 image_points[2] = np.array([x, y], dtype='double')
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 45:
#                 image_points[3] = np.array([x, y], dtype='double')
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 48:
#                 image_points[4] = np.array([x, y], dtype='double')
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 54:
#                 image_points[5] = np.array([x, y], dtype='double')
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             else:
#                 cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

#         # Draw the determinant image points onto the person's face
#         for p in image_points:
#             cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

#         (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

#         cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
#         cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

#         if head_tilt_degree:
#             cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)



#     # Show the frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF

#     # If the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break

# # Do a bit of cleanup
# cv2.destroyAllWindows()
# vs.stop()



# #!/usr/bin/env python
# import cv2
# import dlib
# import numpy as np
# from scipy.spatial import distance as dist
# from imutils.video import VideoStream
# from imutils import face_utils
# import imutils
# import time
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from EAR import eye_aspect_ratio
# from MAR import mouth_aspect_ratio
# from HeadPose import getHeadTiltAndCoords
# import mediapipe as mp

# def create_and_train_model(X, y):
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = Sequential()
#     model.add(Dense(64, input_dim=3, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))

#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
#     score = model.evaluate(X_test, y_test, batch_size=32)
#     print("Test accuracy:", score[1])
    
#     model.save('drowsiness_detection_model.h5')
#     return model, scaler

# # Load the dataset (Assuming you have a dataset of features and labels)
# # For demonstration, we'll generate dummy data
# X = np.random.rand(1000, 3)
# y = np.random.randint(2, size=1000)

# # Create and train the model
# model, scaler = create_and_train_model(X, y)

# # Load the trained model (in real scenarios, load the actual trained model)
# # model = load_model('drowsiness_detection_model.h5')

# # Initialize dlib's face detector and the facial landmark predictor
# print("[INFO] loading facial landmark predictor...")
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# # Initialize the MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# # Initialize the video stream
# print("[INFO] initializing camera...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

# frame_width = 1920
# frame_height = 1080

# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# (mStart, mEnd) = (49, 68)

# EYE_AR_THRESH = 0.25
# MOUTH_AR_THRESH = 0.79
# EYE_AR_CONSEC_FRAMES = 3
# COUNTER = 0

# while True:
#     frame = vs.read()
#     frame = imutils.resize(frame, width=1024, height=576)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     size = gray.shape

#     rects = detector(gray, 0)
    
#     for rect in rects:
#         (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
#         cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)

#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)
#         ear = (leftEAR + rightEAR) / 2.0

#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

#         if ear < EYE_AR_THRESH:
#             COUNTER += 1
#             if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                 cv2.putText(frame, "Eyes Closed!", (500, 20),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         else:
#             COUNTER = 0

#         mouth = shape[mStart:mEnd]
#         mouthMAR = mouth_aspect_ratio(mouth)
#         mar = mouthMAR
#         mouthHull = cv2.convexHull(mouth)
#         cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
#         cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         if mar > MOUTH_AR_THRESH:
#             cv2.putText(frame, "Yawning!", (800, 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         features = np.array([leftEAR, rightEAR, mar])
#         features = scaler.transform(features.reshape(1, -1))
#         prediction = model.predict(features)

#         if prediction > 0.5:
#             cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # MediaPipe Face Mesh for head pose estimation
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             mp.solutions.drawing_utils.draw_landmarks(
#                 image=frame,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

#             # Get the 3D coordinates for head pose estimation
#             image_points = []
#             for landmark in [1, 33, 263, 61, 291, 199]:
#                 x = int(face_landmarks.landmark[landmark].x * frame.shape[1])
#                 y = int(face_landmarks.landmark[landmark].y * frame.shape[0])
#                 image_points.append((x, y))

#             if len(image_points) == 6:
#                 image_points = np.array(image_points, dtype="double")
#                 (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

#                 cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
#                 cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

#                 if head_tilt_degree:
#                     cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

# cv2.destroyAllWindows()
# vs.stop()


# #!/usr/bin/env python
# from scipy.spatial import distance as dist
# from imutils.video import VideoStream
# from imutils import face_utils
# import argparse
# import imutils
# import time
# import dlib
# import math
# import cv2
# import numpy as np
# from EAR import eye_aspect_ratio
# from MAR import mouth_aspect_ratio
# from HeadPose import getHeadTiltAndCoords

# # initialize dlib's face detector (HOG-based) and then create the
# # facial landmark predictor
# print("[INFO] loading facial landmark predictor...")
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# # initialize the video stream and sleep for a bit, allowing the
# # camera sensor to warm up
# print("[INFO] initializing camera...")
# vs = VideoStream(src=0).start()  # تغییر به src=0 برای دوربین پیش‌فرض
# # vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
# time.sleep(2.0)

# # 400x225 to 1024x576
# frame_width = 1920
# frame_height = 1080

# # loop over the frames from the video stream
# # 2D image points. If you change the image, you need to change vector
# image_points = np.array([
#     (359, 391),     # Nose tip 34
#     (399, 561),     # Chin 9
#     (337, 297),     # Left eye left corner 37
#     (513, 301),     # Right eye right corne 46
#     (345, 465),     # Left Mouth corner 49
#     (453, 469)      # Right mouth corner 55
# ], dtype="double")

# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# EYE_AR_THRESH = 0.25
# MOUTH_AR_THRESH = 0.79
# EYE_AR_CONSEC_FRAMES = 3
# COUNTER = 0

# # grab the indexes of the facial landmarks for the mouth
# (mStart, mEnd) = (49, 68)

# while True:
#     # grab the frame from the threaded video stream, resize it to
#     # have a maximum width of 400 pixels, and convert it to
#     # grayscale
#     frame = vs.read()
#     frame = imutils.resize(frame, width=1024, height=576)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     size = gray.shape

#     # detect faces in the grayscale frame
#     rects = detector(gray, 0)

#     # check to see if a face was detected, and if so, draw the total
#     # number of faces on the frame
#     if len(rects) > 0:
#         text = "{} face(s) found".format(len(rects))
#         cv2.putText(frame, text, (10, 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     # loop over the face detections
#     for rect in rects:
#         # compute the bounding box of the face and draw it on the
#         # frame
#         (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
#         cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
#         # determine the facial landmarks for the face region, then
#         # convert the facial landmark (x, y)-coordinates to a NumPy
#         # array
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)

#         # extract the left and right eye coordinates, then use the
#         # coordinates to compute the eye aspect ratio for both eyes
#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)
#         # average the eye aspect ratio together for both eyes
#         ear = (leftEAR + rightEAR) / 2.0

#         # compute the convex hull for the left and right eye, then
#         # visualize each of the eyes
#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

#         # check to see if the eye aspect ratio is below the blink
#         # threshold, and if so, increment the blink frame counter
#         if ear < EYE_AR_THRESH:
#             COUNTER += 1
#             # if the eyes were closed for a sufficient number of times
#             # then show the warning
#             if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                 cv2.putText(frame, "Eyes Closed!", (500, 20),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             # otherwise, the eye aspect ratio is not below the blink
#             # threshold, so reset the counter and alarm
#         else:
#             COUNTER = 0

#         mouth = shape[mStart:mEnd]

#         mouthMAR = mouth_aspect_ratio(mouth)
#         mar = mouthMAR
#         # compute the convex hull for the mouth, then
#         # visualize the mouth
#         mouthHull = cv2.convexHull(mouth)

#         cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
#         cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         # Draw text if mouth is open
#         if mar > MOUTH_AR_THRESH:
#             cv2.putText(frame, "Yawning!", (800, 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


#         # loop over the (x, y)-coordinates for the facial landmarks
#         # and draw each of them
#         for (i, (x, y)) in enumerate(shape):
#             if i == 33:
#                 # something to our key landmarks
#                 # save to our new key point list
#                 # i.e. keypoints = [(i,(x,y))]
#                 image_points[0] = np.array([x, y], dtype='double')
#                 # write on frame in Green
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 8:
#                 # something to our key landmarks
#                 # save to our new key point list
#                 # i.e. keypoints = [(i,(x,y))]
#                 image_points[1] = np.array([x, y], dtype='double')
#                 # write on frame in Green
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 36:
#                 # something to our key landmarks
#                 # save to our new key point list
#                 # i.e. keypoints = [(i,(x,y))]
#                 image_points[2] = np.array([x, y], dtype='double')
#                 # write on frame in Green
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 45:
#                 # something to our key landmarks
#                 # save to our new key point list
#                 # i.e. keypoints = [(i,(x,y))]
#                 image_points[3] = np.array([x, y], dtype='double')
#                 # write on frame in Green
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 48:
#                 # something to our key landmarks
#                 # save to our new key point list
#                 # i.e. keypoints = [(i,(x,y))]
#                 image_points[4] = np.array([x, y], dtype='double')
#                 # write on frame in Green
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             elif i == 54:
#                 # something to our key landmarks
#                 # save to our new key point list
#                 # i.e. keypoints = [(i,(x,y))]
#                 image_points[5] = np.array([x, y], dtype='double')
#                 # write on frame in Green
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#             else:
#                 # everything to all other landmarks
#                 # write on frame in Red
#                 cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
#                 cv2.putText(frame, str(i + 1), (x - 10, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

#         #Draw the determinant image points onto the person's face
#         for p in image_points:
#             cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

#         (head_tilt_degree, start_point, end_point, 
#             end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

#         cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
#         cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

#         if head_tilt_degree:
#             cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     # show the frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF

#     # if the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break

# # do a bit of cleanup
# cv2.destroyAllWindows()
# vs.stop()