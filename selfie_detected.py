import cv2 as cv

# Cascade sınıflandırıcıyı yükle, nesne tanıma algoritması olan CascadeClassifier kullanarak
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Webcam'den video yakalama
capture = cv.VideoCapture(0)

while True:
    # Video'dan bir kare al
    ret, frame = capture.read()

    # Kareyi griye dönüştür
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Yüzleri algıla
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Yüzleri dikdörtgen olarak çiz
    # 153, 0, 204 kodları mor renk için kullanılmış rgb kodlarıdır
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(153, 0, 204),2)

    # Kareyi göster
    cv.imshow('frame',frame)
    if ret:
        cv.imwrite("frame-detected.png", frame)

    # 'q' tuşuna basılınca döngüyü sonlandır
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



# Kaynakları serbest bırak
capture.release()
cv.destroyAllWindows()
