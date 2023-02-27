# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import time

def compute_prediction(nb_hand, predictions) : 
    if nb_hand == 1 : 
        if predictions[0] <6 : 
            return str(predictions[0])
        elif predictions[0] == 6 :
            return '-'
        elif predictions[0] == 7 :
            return '*'
        elif predictions[0] == 8 :
            return '/'
        elif predictions[0] == 9 :
            return '='

    else : 
        if (predictions[0] < 5 and predictions[1]==5) or (predictions[0] == 5  and predictions[1]<5):
            return str(predictions[0] + predictions[1])
        elif predictions[1] == 6  or predictions[1] == 6:
            return '-'
        elif predictions[1] == 7  or predictions[1] == 7:
            return ' '
        elif predictions[1] == 8  or predictions[1] == 8:
            return '/'
        elif predictions[1] == 9  or predictions[1] == 9:
            return '='
        elif predictions[0] == 1 and predictions[1] == 1:
            return '+'


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')
model = pickle.load(open('model_clf.sav', 'rb'))
n = pickle.load(open('Normalizer.sav', 'rb'))

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

formule = ''

# Initialize the webcam
cap = cv2.VideoCapture(0)
t0 = time.time()
is_NewNumber = True
previous = None

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)
    # print(result)
    
    className = ''
    X = []
    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        landmarks_x = np.zeros(21)
        landmarks_y = np.zeros(21)
        for handslms in result.multi_hand_landmarks:
            for i,lm in enumerate(handslms.landmark):
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])
                landmarks_x[i]= int(lmx)                  
                landmarks_y[i]= int(lmy)

            X.append(np.concatenate([landmarks_x, landmarks_y]))
        
                # print(X)
                # print(X.shape)


            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
        X = n.transform(X)
        prediction = model.predict(X)
        # print(prediction)
        classID = prediction
        print(prediction)
        # print(classID)
        current = compute_prediction(len(X), prediction)
        print(current, previous)
        if previous == None:
            previous = current
        elif previous != None and previous != current:
            t0 = time.time()
            print('différent', previous, current)
            previous = current
            # print('différent')
            
        elif current == ' ':
            is_NewNumber = True

        elif time.time()-t0 > 0.50 and (is_NewNumber or True): 
            print('Je suis contetn')
            t0 = time.time()
            print(previous)
            is_NewNumber = False
            if previous != None:
                formule = formule + previous
            else: 
                previous = None
            try : 
                className = formule + className
                formule = formule + className
            except:
                className = prediction
        
        
        
        previous = current


    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(0.05)
    t = time.time()

# release the webcam and destroy all active windows
cap.release()


cv2.destroyAllWindows()