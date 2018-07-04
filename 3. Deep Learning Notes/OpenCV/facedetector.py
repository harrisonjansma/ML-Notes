import sys
import dlib
from skimage import io

predictor_model = "shape_predictor_68_face_landmarks.dat"


file_name = "C://Users//harri//Pictures//harrison.jpg"

face_detector = dlib.get_frontal_face_detector()


image = io.imread(file_name)

detected_faces = face_detector(image,1)

win = dlib.image_window()
win.set_image(image)

for i, face_rect in enumerate(detected_faces):
    win.add_overlap(face_rect)

dlab.hit_enter_to_continue()
