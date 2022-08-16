import base64
import io
import os

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
from flask import Flask, jsonify, request
import json

from Models.Model import dice_loss, combined, dice_coeff
from Utils.segmentation import segment_potatoes

from werkzeug.serving import WSGIRequestHandler

from Utils.utils import get_calibres

app = Flask(__name__)

# Load the necessary keras models
model_mask = keras.models.load_model('Trained Models/mask_yellow.h5',
                                     custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})
model_contour = keras.models.load_model('Trained Models/cnt_yellow.h5',
                                        custom_objects={'combined': combined, 'dice_coeff': dice_coeff})
model_target = keras.models.load_model('Trained Models/targets.h5',
                                       custom_objects={'dice_loss': dice_loss, 'dice_coeff': dice_coeff})

# Initialize pertinent information as global variables
n_pdt, n_targets = 0, 0
heights, diameters = [], []
variety = ''
imgB64 = ''


@app.route("/calibrer", methods=['GET', 'POST'])
def calibRoute():
    global n_pdt, n_targets, imgB64, heights, diameters, variety
    if request.method == 'POST':
        request_data = request.data
        request_data = json.loads(request_data.decode('utf-8'))
        base64img = request_data['base64img']
        variety = request_data['variety']
        img = base64.b64decode(base64img)
        pil_img = Image.open(io.BytesIO(img))
        seg_img, diameters, heights, n_targets = segment_potatoes(pil_img, model_mask, model_contour, model_target,
                                                                  patch_size=256, resize=(2048, 1536), norm_fact=255,
                                                                  dev_mode=False)
        seg_img = Image.fromarray(cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)).resize((576, 432))
        seg_img.save('res.jpg')

        with open("res.jpg", "rb") as img_file:
            imgB64 = base64.b64encode(img_file.read()).decode('ascii')
        os.remove('res.jpg')
        return ""
    else:
        return jsonify({'n_pdt': len(list(diameters)),
                        'n_targets': n_targets,
                        'img': imgB64})


@app.route("/statistics", methods=['GET'])
def statsRoute():
    global n_pdt, heights, diameters, variety
    assert variety in ['Burbank', 'Mountain Gem']

    # Compute specific statistics
    GT_4po = int((np.array(heights) > 4 * 25.4).sum())
    GT_1_7po = int((np.array(diameters) > 1.7 * 25.4).sum())

    # Predict the weights
    if variety == 'Burbank':
        d_norm = [(i - 39) / (84 - 39) for i in diameters]
        h_norm = [(i - 62) / (179 - 62) for i in heights]
        variety_pdt = 'burbank'
    else:
        d_norm = [(i - 39) / (81 - 39) for i in diameters]
        h_norm = [(i - 48) / (190 - 48) for i in heights]
        variety_pdt = 'mountain_gem'

    model = keras.models.load_model(f"Trained models/{variety_pdt}_weight.h5")
    sample = np.stack([h_norm, d_norm], axis=-1)
    weights = model.predict(sample)
    if request.method == 'GET':
        return jsonify({'n_pdt': len(list(diameters)),
                        'heights': list(heights),
                        'diameters': list(diameters),
                        'weights': [float(i) for i in np.squeeze(weights)],
                        'totalWeight': int(np.sum(weights)),
                        'h > 4po': GT_4po,
                        'd > 1.7po': GT_1_7po})


@app.route("/calibres", methods=['GET'])
def calRoute():
    global diameters
    if request.method == 'GET':
        return jsonify(get_calibres(diameters))


if __name__ == '__main__':
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(debug=False, ssl_context='adhoc')
