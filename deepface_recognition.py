import cv2
from deepface import DeepFace


def deepface_analyze(filepath):
    image = cv2.imread(filepath)
    analyze = DeepFace.analyze(image, enforce_detection=False)
    emotion = analyze['dominant_emotion']
    age = analyze['age']
    gender = analyze['gender']
    race = analyze['dominant_race']
    return f'{emotion} {age} year old {race} {gender}'
