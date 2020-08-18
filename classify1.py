import os
import sys
import cv2
import numpy as np
from data import DataSet
from extractor import Extractor
from keras.models import load_model
import tensorflow as tf
import  time


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()




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

model_path = 'frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.7



capture = cv2.VideoCapture("V_101.mp4")
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
seq_length = 5
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("result.mp4", fourcc, 15, (int(width), int(height)))

# Get the dataset.
data = DataSet(seq_length=41, class_limit=2, image_shape=(height, width, 3))
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

extract_model = Extractor(image_shape=(height, width, 3))


saved_LSTM_model = tf.keras.models.load_model(
    "F:\\BE Project work\\LSTM-video-classification-master\\data\\checkpoints\\lstm-features.022-0.035.hdf5",
    custom_objects=None,
    compile=False
)
# get the model.
#saved_LSTM_model = load_model("lstm-features.hdf5")
print(capture)
frames = []
frame_count = 0
start = time.time()
human = False
hcount = 0
while True:
    ret, frame = capture.read()
    # Bail out when the video file ends
    if not ret:
        break
    # Save each frame of the video to a list
    

    boxes, scores, classes, num = odapi.processFrame(frame)
    
    if frame_count == 0 :
      human = False
      for i in range(len(boxes)):
              # Class 1 represents human
              if classes[i] == 1 and scores[i] > threshold:
                  human = True
                  break

    if human == False :
      continue

    frame_count += 1
    frames.append(frame)

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
        #cv2.imshow("Frame",image)
       
        #if  cv2.waitKey(1) and 0xFF == ord('q'):
        #   cv2.destroyAllWindows()
        #    break
        video_writer.write(image)

    frames = []
print("time req--".format(time.time()-start))
video_writer.release()
