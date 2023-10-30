
## ASLingual

### Overview

ASLingual is a deep learning software powered by Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). Using Python's TensorFlow framework, ASLingual is trained with several datasets of ASL (American Sign Language) signs and sequences to provide accurate, real-time captions.

ASLingual uses the spatial recognition capabilities of CNNs to identify individual sign gestures accurately. Each sign is converted into a high-dimensional vector space, allowing the model to discern even subtle differences in gestures.

### Features

1. **Real-time Sign Recognition**  
ASLingual captures and processes video feed in real-time, translating ASL signs into textual captions.

```python
import cv2
import time
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
```

2. **Training on Custom Datasets**  
Users can train ASLingual on their custom datasets, enhancing its recognition capabilities.

```python
import tensorflow as tf

def train_on_dataset(dataset_path):
    data = tf.data.Dataset.from_tensor_slices(tf.keras.preprocessing.image.load_img(dataset_path))
    model = tf.keras.Sequential([path])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(data, epochs=10)
```

3. **Gesture Recognition with MediaPipe**  
ASLingual uses MediaPipe to recognize hand gestures in real-time, leveraging the spatial recognition capabilities of CNNs.

```python
gesture_model = HandGestureModel()

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        gesture = gesture_model.predict(hand_landmarks.landmark)
        cv2.putText(frame, gesture, (10, 60), FONT, FONT_SCALE, THICKNESS, cv2.LINE_AA)
```

### Running ASLingual Locally

1. **Clone the repository**:

```bash
git clone https://github.com/wxlkda/aslingual.git
cd aslingual
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the application**:

```bash
python main.py
```

