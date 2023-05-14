from flask import Flask, request
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    # 处理跨域请求
    origin = request.headers.get('Origin')
    if origin is not None:
        response_headers = {
            'Access-Control-Allow-Origin': origin,
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Authorization, Content-Type',
            'Access-Control-Max-Age': '86400',
        }
        if request.method == 'OPTIONS':
            return '', 204, response_headers

        # 保存文件到指定目录
        file = request.files['file']
        filename = file.filename
        save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', filename)
        file.save(save_path)
        # 打印日志到文件
        with open('app.log', 'a') as f:
            f.write('上传文件：%s\n' % filename)
        return {'msg': '上传成功'}


@app.route('/start', methods='get')
def getProgress():
    t = request.args.get('type', type=str)
    # unzip the file

    if t == "music":
        # start music
        pass
    else:
        # start pic
        pass


@app.route('/progress', methods='get')
def getProgress():
    t = request.args.get('type', type=str)
    if t == "music":
        from . import g_progress
        return g_progress
    else:
        from . import g_progress
        return g_progress


if __name__ == '__main__':
    app.run()