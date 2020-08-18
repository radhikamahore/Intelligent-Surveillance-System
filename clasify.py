import time
start = time.time()


import os
import sys
import cv2
import numpy as np
from data import DataSet
from extractor import Extractor
from keras.models import load_model
print('time take to load imports {:0.3f}'.format(time.time() - start))

'''if (len(sys.argv) == 5):
    seq_length = int(sys.argv[1])
    class_limit = int(sys.argv[2])
    saved_model = sys.argv[3]
    video_file = sys.argv[4]
else:
    print ("Usage: python clasify.py sequence_length class_limit saved_model_name video_file_name")
    print ("Example: python clasify.py 75 2 lstm-features.095-0.090.hdf5 some_video.mp4")
    exit (1)
'''
capture = cv2.VideoCapture("newfi73.avi")
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) )  # float
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
seq_length = 5
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("result.avi", fourcc, 15, (int(width), int(height)))

# Get the dataset.
data = DataSet(seq_length=41, class_limit=2, image_shape=(240, 320, 3))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# get the model.
print("Loading Model .......")
extract_model = Extractor(image_shape=(240,320, 3))
saved_LSTM_model = load_model("data\\checkpoints\\lstm-features.022-0.035.hdf5",compile='False')
print(capture)
print("Captured Video....")
print("Model Loaded.......")
print('time take to load model {:0.3f}'.format(time.time() - start))

frames = []
frame_count = 0
while True:
    ret, frame = capture.read()
    # Bail out when the video file ends
    if not ret:
        break
    # Save each frame of the video to a list
    frame_count += 1
    image1=cv2.resize(frame,(320,240))
    frames.append(image1)

    if frame_count < seq_length:
        continue # capture frames untill you get the required number for sequence
    else:
        frame_count = 0

    # For each frame extract feature and prepare it for classification
    sequence = []
    for image in frames:
        features = extract_model.extract_image(image)
        #print(features)
        sequence.append(features)
      
    # Clasify sequence
    prediction = saved_LSTM_model.predict(np.expand_dims(sequence, axis=0))
    print(prediction)
    values = data.print_class_from_prediction(np.squeeze(prediction, axis=0))

    # Add prediction to frames and write them to new video
    for image in frames:
        for i in range(len(values)):
            cv2.putText(image, values[i], (40, 40 * i + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.imshow("Frame",image)
       
        if  cv2.waitKey(1) and 0xFF == ord('q'):
           break
        video_writer.write(image)

    frames = []
cv2.destroyAllWindows()

video_writer.release()
