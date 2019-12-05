import json

from flask import Flask, render_template, request, abort, jsonify

from testing.test_conv import test_cnn
from testing.test_cnn_four_finger import test_cnn_four_finger
from testing.predict_from_image import predict_from_image

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/four_finger")
def four_finger():
    return render_template('four_finger.html')


@app.route("/testModel/", methods=['POST'])
def test_model():
    result = 'This works!'
    nn_type = request.form['nn_type']
    data = request.form['test_data']

    score = None
    plot_src = None
    title = None
    if nn_type == 'cnn':
        score, title = test_cnn(data)
        plot_src = '../static/cnn_performance.png'
    # elif nn_type is 'rnn':
    #     rnn_test(data)

    return render_template('index.html', score=score, plot_src=plot_src, title=title)


@app.route("/testFourFinger/", methods=['POST'])
def test_four_finger_model():
    model = request.form['model']
    data = request.form['test_data']

    errors, title = test_cnn_four_finger(data, model)
    plot_srcs = ['../static/four_finger/index.png', '../static/four_finger/middle.png', '../static/four_finger/ring.png',
                '../static/four_finger/pinky.png']
    errors = ['Index error: ' + str(errors[0]), 'Middle error: ' + str(errors[1]), 'Ring error: ' + str(errors[2]),
              'Pinky error: ' + str(errors[3])]

    return render_template('four_finger.html', error_index=errors[0], error_middle=errors[1], error_ring=errors[2],
                           error_pinky=errors[3], plot_src_index=plot_srcs[0], plot_src_middle=plot_srcs[1],
                           plot_src_ring=plot_srcs[2], plot_src_pinky=plot_srcs[3], title=title)


@app.route('/api/predictAngles', methods=['POST'])
def predict():
    image = request.json["image"]
    angles = predict_from_image(image)
    return "JSON Message: " + json.dumps({"angles": angles})


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
