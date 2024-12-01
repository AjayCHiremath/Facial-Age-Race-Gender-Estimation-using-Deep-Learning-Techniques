import cv2
import test
import sys
import predictor
# import imutils
# import dlib

# detector = dlib.get_frontal_face_detector()
# def convert_and_trim_bb(image, rect):
# 	# extract the starting and ending (x, y)-coordinates of the
# 	# bounding box
# 	startX = rect.left()
# 	startY = rect.top()
# 	endX = rect.right()
# 	endY = rect.bottom()
# 	# ensure the bounding box coordinates fall within the spatial
# 	# dimensions of the image
# 	startX = max(0, startX)
# 	startY = max(0, startY)
# 	endX = min(endX, image.shape[1])
# 	endY = min(endY, image.shape[0])
# 	# compute the width and height of the bounding box
# 	w = endX - startX
# 	h = endY - startY
# 	# return our bounding box coordinates
# 	return (startX, startY, w, h)

# def get_faces_dlib(image):
#     # load dlib's HOG + Linear SVM face detector
#     print("[INFO] loading HOG + Linear SVM face detector...")
#     detector = dlib.get_frontal_face_detector()
#     # load the input image from disk, resize it, and convert it from
#     # BGR to RGB channel ordering (which is what dlib expects)
#     # image = cv2.imread(args["image"])
#     image = imutils.resize(image, width=600)
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # perform face detection using dlib's face detector
#     rects = detector(rgb, 1)

#     boxes = [convert_and_trim_bb(image, r) for r in rects]
#     # loop over the bounding boxes
#     for (x, y, w, h) in boxes:
#         # draw the bounding box on our image
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Get user supplied values
import face_recognition
def getFacesDLIB(frame, get_encoding=False):
    face_locations = face_recognition.face_locations(frame)
    if get_encoding:
        face_encodings = face_recognition.face_encodings(frame, face_locations)
    # for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        for (top, right, bottom, left),encoding in zip(face_locations,face_encodings):
            x,y,w,h = left,top,right-left,bottom-top
            face = frame[y:y+h, x:x+w]
            yield (x,y,w,h),encoding
    else:
        for top, right, bottom, left in face_locations:
            x,y,w,h = left,top,right-left,bottom-top
            face = frame[y:y+h, x:x+w]
            yield x,y,w,h

        # print(face.shape)
        # predictor.predict(face)


def getFaces(image):        
    cascPath = "haarcascade_frontalface_alt2.xml"
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    # Read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(3, 3)
    )

    print ("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for face in faces:
        yield face
        # (x,y,w,h) = face
        # predictor.predict(image[y-10:y+h+10,x-10:x+w+10])
    #     cv2.waitKey(0)
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.imshow("Faces found", image)
    # cv2.waitKey(0)
if __name__ == "__main__":
    import numpy as np
    from urllib.request import urlopen
    url = 'https://direct.rhapsody.com/imageserver/images/alb.547910504/500x500.jpg'
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    im = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # im = cv2.imread("shahrukh.jpg")
    getFacesDLIB(im)
