import tensorflow.keras
import cv2
import numpy as np
import pyfirmata                        
import time     

board = pyfirmata.Arduino('COM6') 

def turnOnGreen():
    board.digital[13].write(1)
    print ("Green LED turned ON")     
                    
    
def turnOffGreen():
    board.digital[13].write(0)
    print ("Green LED turned OFF")     


def turnOnRed():
    board.digital[12].write(1)
    print ("Red LED turned ON")     
              
    
def turnOffRed():
    board.digital[12].write(0)
    print ("Red LED turned OFF")     

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

#Load the labels
with open('labels.txt', 'r') as f:
    class_names = f.read().split('\n')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

size = (224, 224)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    start = time.time()

    ret, img = cap.read()

    height, width, channels = img.shape

    scale_value = width/height

    img_resized = cv2. resize(img, size, fx=scale_value, fy=1, interpolation=cv2.INTER_NEAREST)

    # Turn the image into a numpy array
    img_array = np.asarray(img_resized)

    # Normalize the image
    normalized_img_array = (img_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_img_array

    # Run the inference
    prediction = model.predict(data)
    #print(prediction)

    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    #print("Class: ", class_name)
    #print("Confidence score: ", confidence_score)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    temp = class_name
    #print("FPS: ", fps)
    #cv2.putText(img, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    #print(float("{:.2f}".format(confidence_score*100)))
    #print(class_name)
    if(class_name == "0 Thumbs Up" and float("{:.2f}".format(confidence_score*100))>float(80)):
        turnOnGreen()
        turnOffRed()
        print("Thumbs Up!")
        time.sleep(5)

    elif(class_name == "1 Thumbs Down" and float("{:.2f}".format(confidence_score*100))>float(99)):
        turnOnRed()
        turnOffGreen()
        print("Thumbs Down!")
        time.sleep(5)

    #print(class_name)
    #print(str(float("{:.2f}".format(confidence_score*100))) + "%")
    #cv2.putText(img, class_name, (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    #cv2.putText(img, str(float("{:.2f}".format(confidence_score*100))) + "%", (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    #cv2.imshow('Classification Resized', img_resized)
    #cv2.imshow('Classification Original', img)


    #****press the esc to stop the video capture window
    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()






