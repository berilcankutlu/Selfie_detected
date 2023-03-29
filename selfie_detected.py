import cv2 as cv

# Install Cascade classifier, using CascadeClassifier, object recognition algorithm
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from webcam
capture = cv.VideoCapture(0)

while True:
    # Get a frame from video
    ret, frame = capture.read()

    # Convert square to gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw faces as rectangles
    # 153, 0, 204 codes are rgb codes used for purple color
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(153, 0, 204),2)

    # Show frame
    cv.imshow('frame',frame)
    if ret:
        cv.imwrite("frame-detected.png", frame)

    # End loop when 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break




capture.release()
cv.destroyAllWindows()
