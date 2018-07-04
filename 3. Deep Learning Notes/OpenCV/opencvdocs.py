import cv2
import sys
import dlib
from skimage import io


predictor_model = "C:\Users\harri\Documents\GitHub\ML-Notes\3. Deep Learning Notes\OpenCV"

file_name = sys.argv[1]

face_detector = dlib.get_frontal_face_detector()
