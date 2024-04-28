import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# Load the trained model
model = load_model(r"C:\Users\prisha anand\Desktop\tf model\model_01.h5")

# Mapping numerical labels to class names
class_names = {0: 'battery', 1: 'biological', 2: 'cardboard', 3: 'clothes', 4: 'glass', 
               5: 'metal', 6: 'paper', 7: 'plastic', 8: 'shoes', 9: 'trash'}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set dimensions for resizing the image
width, height = 300, 300

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame
    frame_resized = cv2.resize(frame, (width, height))
    
    # Preprocess the frame
    img_array = image.img_to_array(frame_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    # Get the class name
    class_name = class_names.get(predicted_class, "Unknown")
    
    # Display the class name
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, class_name, (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Webcam', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
