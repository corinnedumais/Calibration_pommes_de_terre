import numpy as np
from flask import Flask, render_template, request
from tensorflow import keras

app = Flask(__name__)


@app.route("/")
def index():
    if request.args.get('width', None) and request.args.get('height', None):
        w = float(request.args.get('width', None))
        h = float(request.args.get('height', None))
        variety = request.args.get('variety', None)

        if variety == 'mountain_gem':
            w = (w - 39) / (81 - 39)
            h = (h - 48) / (190 - 48)
        else:
            w = (w - 39)/(84 - 39)
            h = (h - 62)/(179 - 62)

        model = keras.models.load_model(f"../Trained models/{variety}_weight.h5")
        sample = np.expand_dims(np.array([h, w]), 0)
        weight = str(int(model.predict(sample)[0])) + ' g'

    else:
        weight = ''

    return render_template("index2.html", weight=weight)


if __name__ == '__main__':
    app.run(debug=True)
