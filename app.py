import base64
import io

from PIL import Image
from tensorflow import keras
from flask import Flask, jsonify, request
import json

from Models.Model import dice_loss, combined, dice_coeff
from Utils.segmentation import segment_potatoes

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
seg_img = ''


@app.route("/", methods=['GET', 'POST'])
def calibRoute():
    global n_pdt, n_targets, seg_img, heights, diameters
    if request.method == 'POST':
        request_data = request.data
        request_data = json.loads(request_data.decode('utf-8'))
        base64img = request_data['base64img']
        img = base64.b64decode(base64img)
        pil_img = Image.open(io.BytesIO(img))
        seg_img, diameters, heights, n_targets = segment_potatoes(pil_img, model_mask, model_contour, model_target,
                                                                  patch_size=256, resize=(2048, 1536), norm_fact=255,
                                                                  dev_mode=False)
        return ""
    else:
        return jsonify({'n_pdt': len(list(diameters)),
                        'n_targets': n_targets,
                        'diameters': json.dumps(list(diameters)),
                        'heights': json.dumps(list(heights)),
                        'img': str(base64.b64encode(b'seg_img'))})


if __name__ == '__main__':
    app.run(debug=True)
