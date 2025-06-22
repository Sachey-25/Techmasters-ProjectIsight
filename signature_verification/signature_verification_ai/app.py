import os
import numpy as np
from flask import Flask, render_template, request

from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.layers import Input

from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocess import preprocess_image

UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model and intermediate model
model = load_model('model/signature_cnn.h5')

# Rebuild Sequential model with known input shape
input_layer = Input(shape=(150, 220, 1))  # use the exact shape used during training

# Pass layers manually from saved model
x = input_layer
for layer in model.layers[:-1]:  # exclude final classification layer
    x = layer(x)



#model.build((None, 100, 100, 1))  # Ensure input shape matches your preprocess
intermediate_model = Model(inputs=input_layer, outputs=x)

def predict_signature(img_path):
    img = preprocess_image(img_path)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    return f"{float(pred):.4f}", "✅ Genuine" if pred > 0.5 else "❌ Forged"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None
    result = None
    similarity_score = None
    filenames = []

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            confidence, prediction = predict_signature(filepath)

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           filename=filename,
                           result=result,
                           similarity=similarity_score,
                           filenames=filenames)

@app.route('/compare', methods=['POST'])
def compare():
    result = None
    similarity_score = None
    filenames = []

    file1 = request.files.get('file1')
    file2 = request.files.get('file2')

    if file1 and file2:
        filename1 = file1.filename
        filename2 = file2.filename

        path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

        file1.save(path1)
        file2.save(path2)

        filenames = [filename1, filename2]

        img1 = preprocess_image(path1)
        img2 = preprocess_image(path2)

        emb1 = intermediate_model.predict(np.expand_dims(img1, axis=0))[0]
        emb2 = intermediate_model.predict(np.expand_dims(img2, axis=0))[0]

        similarity_score = cosine_similarity([emb1], [emb2])[0][0]
        result = "✅ Match" if similarity_score > 0.75 else "❌ Mismatch"

    return render_template('index.html',
                           prediction=None,
                           confidence=None,
                           filename=None,
                           result=result,
                           similarity=similarity_score,
                           filenames=filenames)

if __name__ == '__main__':
    app.run(debug=True)
