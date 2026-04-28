from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
import os

app = Flask(__name__)

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model("sign_model.h5")

# ===============================
# CLASS NAMES
# ===============================
class_names = [
    '0_GHA','10_RA','1_GA','2_KA','3_KAA',
    '4_NGA','5_CHA','6_CHAA','7_BA','8_PA','9_LA'
]

# ===============================
# MALAYALAM MAP
# ===============================
malayalam_map = {
    '0_GHA': 'ഘ','1_GA': 'ഗ','2_KA': 'ക','3_KAA': 'ഖ',
    '4_NGA': 'ങ','5_CHA': 'ച','6_CHAA':'ഛ',
    '7_BA':'ബ','8_PA':'പ','9_LA':'ല','10_RA':'റ'
}

# ===============================
# FONT
# ===============================
font = ImageFont.truetype("nirmala.ttf", 80)

# ===============================
# MEDIAPIPE
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

camera = None

# ===============================
# ROUTES
# ===============================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/camera')
def camera_page():
    return render_template('camera.html')

# ===============================
#  UPLOAD ROUTE (FIXED)
# ===============================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        if 'file' not in request.files:
            return "No file selected"

        file = request.files['file']

        if file.filename == '':
            return "No file selected"

        # save image
        filepath = os.path.join("static", "upload.jpg")
        file.save(filepath)

        # read image
        img = cv2.imread(filepath)

        if img is None:
            return "Image not found or invalid"

        # preprocess
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # predict
        prediction = model.predict(img)
        index = np.argmax(prediction)

        eng_label = class_names[index]
        mal_label = malayalam_map.get(eng_label, eng_label)

        return render_template("result.html", label=mal_label)

    return render_template("upload.html")

# ===============================
# CAMERA START
# ===============================
@app.route('/start')
def start():
    global camera
    camera = cv2.VideoCapture(0)
    return "Camera Started"

# ===============================
# VIDEO STREAM
# ===============================
def gen_frames():
    global camera

    while True:
        if camera is None:
            break

        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        display_text = ""

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

                padding = 80

                x1 = max(0, min(x_list) - padding)
                y1 = max(0, min(y_list) - padding)
                x2 = min(w, max(x_list) + padding)
                y2 = min(h, max(y_list) + padding)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                roi = frame[y1:y2, x1:x2]

                if roi.size != 0:
                    roi = cv2.resize(roi, (64, 64))
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi = roi / 255.0
                    roi = np.expand_dims(roi, axis=0)

                    prediction = model.predict(roi, verbose=0)
                    index = np.argmax(prediction)

                    eng_label = class_names[index]
                    display_text = malayalam_map.get(eng_label, eng_label)

        # Malayalam text
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img_pil)
        draw.text((50, 50), display_text, font=font, fill=(255, 0, 0))

        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ===============================
# VIDEO FEED
# ===============================
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ===============================
# STOP CAMERA
# ===============================
@app.route('/stop')
def stop():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return "Camera Stopped"

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True)