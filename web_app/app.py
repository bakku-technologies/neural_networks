from flask import Flask, render_template, request

from testing.test_conv import test_cnn

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route("/")
def hello():
    return render_template('index.html')


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


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response


app.run(host='0.0.0.0', port=5000)
