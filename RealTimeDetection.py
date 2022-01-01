# !pip install opencv-python
# !pip install opencv-python-headless
import tensorflow as tf
import cv2
import numpy as np

batch_size = 32

def l2_loss(y_true, y_pred):
  x_diff =  y_true[0:14] - y_pred[0:14]
  y_diff =  y_true[14:28] - y_pred[14:28]
  diff = tf.math.pow(x_diff,2) + tf.math.pow(y_diff,2)
  return tf.keras.backend.sum(diff)/batch_size

model = tf.keras.models.load_model('exported_model', custom_objects={'l2_loss': l2_loss})

W = 202


cam = cv2.VideoCapture(0)

if not cam.isOpened():
  print ("Could not open cam")
  exit()

while(1):
    ret, frame = cam.read()
    if ret:
        
        frame = cv2.flip(frame,1)        
        ROI_frame = frame[0:1200, 0:1200].copy()
        print(ROI_frame.shape)
        ROI_small = cv2.resize(ROI_frame, (W ,W))
        ROI_small = ROI_small[None,:,:,:]/255.
        print(ROI_small.shape)
        out = model.predict(ROI_small)*W
        print(out)
        disp = cv2.resize(ROI_frame, (W ,W))
        for i in range(14):
          disp = cv2.circle(disp, (out[0][i],out[0][i+13]), 5, (0,255,0))
        cv2.imshow('Roi', disp)
        
        
        print()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()