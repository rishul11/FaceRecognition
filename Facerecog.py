import cv2
import face_recognition
import numpy as np

# Load and encode known face images
known_image = face_recognition.load_image_file("known_face.jpg")  # Replace with your image
known_encoding = face_recognition.face_encodings(known_image)[0]

# List of known faces and their names
known_encodings = [known_encoding]
known_names = ["Your Name"]  # Replace with the person's name

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # Check if a match is found
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # Get face coordinates
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Face Recognition", frame)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
