from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def uploadFile():
    if request.method == 'POST':
        uploadedFile = request.files['file']
        if uploadedFile:
            tempDir = os.path.join(os.path.dirname(__file__), 'temp')
            if not os.path.exists(tempDir):
                os.makedirs(tempDir)
            uploadedFile.save(os.path.join(tempDir, uploadedFile.filename))
            return 'File uploaded successfully!'
        else:
            return 'No file uploaded'

if __name__ == '__main__':
    app.run(debug=True)
