from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from keras.models import load_model
from base64 import b64encode

app = Flask(__name__)

IMG_SIZE = 128

# Define the path to your model file 
MODEL_PATH = 'D:\projects\Violence\Project Phase II\VideoViolenceDetection\model.h5'

def detect_violence(video):
    if not os.path.exists('output'):
        os.mkdir('output')

    print("Loading model ...")
    model = load_model(MODEL_PATH)

    vs = cv2.VideoCapture(video)
    (W, H) = (None, None)
    count = 0
    last_processed_time = 0

    results = []

    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        try:
            count += 1
            timestamp = vs.get(cv2.CAP_PROP_POS_MSEC) / 1000  

            # Process one frame per second
            if timestamp - last_processed_time >= 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)).astype("float16") / 255
                frame_input = np.expand_dims(frame_resized, axis=0)

                # Predict violence
                preds = model.predict(frame_input)[0]
                label = (preds > 0.6)[0]  # Adjust the threshold as needed

                if label:
                    result = f"Violence detected at timestamp {timestamp} (Video Time: {format_time(timestamp)})"
                    results.append(result)
                    print(result)  

                last_processed_time = timestamp

        except Exception as e:
            print("Error:", e)
            break

    print("Cleaning up...")
    vs.release()

    return results



def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    video_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            
            results = detect_violence(filepath)
            video_url = '/' + filepath  

            return render_template('index.html', message='File uploaded successfully', results=results, video_url=video_url)

    return render_template('index.html', video_url=video_url)



if __name__ == '__main__':
    app.run(debug=True)
