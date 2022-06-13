from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename
from predict import predict

app = Flask(__name__, static_folder='staticFiles')


UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'


filename = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']

        # Extracting uploaded data file name
        data_filename = secure_filename('PREDICT.csv')

        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(
            app.config['UPLOAD_FOLDER'], data_filename))

        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(
            app.config['UPLOAD_FOLDER'], data_filename)

    models = os.listdir('./Models')
    return render_template('prep.html', models=models)


@app.route('/predict', methods=['POST'])
def predict_price():

    model_name = request.form['model']
    result = predict(model_name)
    verdict = None
    if result > 0.5:
        verdict = 'Bullish'
    else:
        verdict = 'Bearish'
    return render_template('result.html', verdict=verdict, result=result, model=model_name)


if __name__ == '__main__':
    app.run()
