import threading
import zipfile

from flask import Flask, request
from flask_cors import CORS
import os

import train
from train import Train

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
        # # 判断文件是否为.zip文件
        # if not zipfile.is_zipfile(save_path):
        #     return {'code': 400, 'msg': '文件不是zip压缩文件格式', 'data': None}

        # 解压缩文件并列出各级文件
        with zipfile.ZipFile(save_path, 'r') as zip_file:
            file_list = zip_file.namelist()
            return {'code': 200, 'msg': '上传成功', 'data': file_list}


trainInstance = None


@app.route('/start', methods=["GET"])
def startTask():
    task = request.args.get('type', type=str)
    if task not in ['music', 'labelme']:
        return {'code': 400, 'msg': 'type error', 'data': None}
    global trainInstance
    trainInstance = Train()
    threading.Thread(target=trainInstance.train).start()
    return {'code': 200, 'msg': 'task begin', 'data': None}


@app.route('/getProgress', methods=["GET"])
def getProgress():
    global trainInstance
    if type(trainInstance) == train.Train:
        s, c, a, ll = trainInstance.getProgress()
        print(s, c, a, ll)
        if s:
            return {'code': 200, 'msg': 'task progress', 'data': {'current': c, 'epochs': a, 'logs': ll}}
    return {'code': 400, 'msg': 'get progress error', 'data': None}


if __name__ == '__main__':
    app.run(processes=True)
    # app.run(threaded=True)