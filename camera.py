import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("sign_model.h5")

class_names = ['0_GHA','1_GA','2_KA','3_KAA','4_NGA','5_CHA','6_CHAA','7_BA','8_PA','9_LA','10_RA']


def run_camera():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not opening")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        img = cv2.resize(frame, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        class_index = np.argmax(prediction)

        label = class_names[class_index]

        # show text on screen
        cv2.putText(frame, label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Sign Language Camera", frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()