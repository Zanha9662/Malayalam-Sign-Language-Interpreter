import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
from collections import deque, Counter

# Load trained model
model = load_model("sign_model.h5")

#  Class names
class_names = ['0_GHA', '1_GA', '2_KA', '3_KAA', '4_NGA',
               '5_CHA','6_CHAA','7_BA','8_PA','9_LA','10_RA']

#  Malayalam mapping
malayalam_map = {
    '0_GHA': 'ഘ',
    '1_GA': 'ഗ',
    '2_KA': 'ക',
    '3_KAA': 'ഖ',
    '4_NGA': 'ങ',
    '5_CHA': 'ച',
    '6_CHAA':'ഛ',
    '7_BA':'ബ',
    '8_PA':'പ',
    '9_LA':'ല',
    '10_RA':'റ'
}

#  Load Malayalam font
font = ImageFont.truetype("nirmala.ttf", 80)

#  Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working")
    exit()

print("Camera started... Press Q to exit")

#  Prediction smoothing
prediction_buffer = deque(maxlen=5)

#  Confidence threshold
confidence_threshold = 0.8

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI box
    x, y, w, h = 100, 100, 300, 300
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    hand = frame[y:y+h, x:x+w]

    # Blur (noise reduce)
    hand = cv2.GaussianBlur(hand, (5,5), 0)

    # Preprocess
    img = cv2.resize(hand, (64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Apply confidence filter
    if confidence > confidence_threshold:
        prediction_buffer.append(class_index)

        if len(prediction_buffer) == 5:
            most_common = Counter(prediction_buffer).most_common(1)[0][0]
            label = class_names[most_common]
            mal_text = malayalam_map[label]
        else:
            mal_text = "..."
    else:
        mal_text = "..."

    # Convert OpenCV → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)

    # Draw Malayalam letter
    draw.text((50, 50), mal_text, font=font, fill=(255, 0, 0))

    # Convert back RGB → OpenCV
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Sign Language Prediction", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()