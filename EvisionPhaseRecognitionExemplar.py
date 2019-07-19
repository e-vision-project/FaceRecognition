import os, glob
import face_recognition
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

image_dir = '/home/kostasgeorgiadis/sync/Deliverable/Train/'
os.chdir(image_dir)
Train_encoding=[]
filenames=[]
images_ALL=[]
for file in glob.glob('*.jpg'):
    image = face_recognition.load_image_file(image_dir+file)
    face_loc = face_recognition.face_locations(image)
    if len(face_loc)==1:
        faces = image[face_loc[0][0]:face_loc[0][2], face_loc[0][3]:face_loc[0][1]]
        images_ALL.append(image)
        filenames.append(file)
        Train_encoding.append((face_recognition.face_encodings(faces)[0]))

test_dir = '/home/kostasgeorgiadis/sync/Deliverable/Test/'
test_img_ID = 'GOPR0203.JPG'
image_test = face_recognition.load_image_file(test_dir+test_img_ID)
face_loc_test = face_recognition.face_locations(image_test)
faces_test = image_test[face_loc_test[0][0]:face_loc_test[0][2], face_loc_test[0][3]:face_loc_test[0][1]]
Test_encoding = face_recognition.face_encodings(faces_test)[0]

distances_true = []
results = face_recognition.compare_faces(Test_encoding,Train_encoding)
labels_true = [i for i,x in enumerate(results) if x==True] # => [1, 3]
for l in range(len(labels_true)):
    distances_true.append(np.sqrt(np.sum(np.square(Test_encoding-Train_encoding[labels_true[l]]))))
    print("Distance between Input image and image", filenames[labels_true[l]], "is", distances_true[l])
    plt.figure()
    plt.imshow(images_ALL[labels_true[l]])