import tensorflow as tf
import cv2
import numpy as np

TF_MODEL_FILE_PATH = 'model.tflite'  # The default path to the saved TensorFlow Lite model
class_names = ['Cat', 'Dog', 'Racoon']
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

print(interpreter.get_signature_list())
print(interpreter.get_input_details())

classify_lite = interpreter.get_signature_runner('serving_default')

cam = cv2.VideoCapture(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

cv2.namedWindow("Test")
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    # frame = cv2.imread("C:\\Users\\alber\\Documents\\Gits\\ProyectoFinalVision\\dataset\\Dog\\16.jpg")

    frame_resized = cv2.resize(frame, (128, 128))
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    rgb_float = np.float32(rgb)
    float_frame = tf.expand_dims(rgb_float, 0)

    predictions_lite = classify_lite(sequential_input=float_frame)['outputs']

    score_lite = tf.nn.softmax(predictions_lite)
    message = ""

    if np.max(score_lite) < 0.5:
        message = "None"
    else:
        message = "%d %s" % (100 * np.max(score_lite), class_names[np.argmax(score_lite)])

    print(message)

    cv2.putText(frame, message, (0, 200), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("Test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()
cv2.destroyAllWindows()
