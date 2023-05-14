from flask import Flask, request
import logging

app = Flask(__name__)

# Allow cross-origin requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Upload endpoint
@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload():
    if request.method == 'OPTIONS':
        return ''
    else:
        file = request.files['file']
        file.save('../data/' + file.filename)
        app.logger.info('File saved: %s', file.filename)
        return 'File uploaded successfully'

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

if __name__ == '__main__':
    app.run()
