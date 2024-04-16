from flask import Flask, render_template, send_from_directory, make_response
import os

app = Flask(__name__)
DETECTED_FRAMES_DIR = os.path.abspath('detected_frames')  # 使用绝对路径更安全

@app.route('/')
def index():
    # 动态生成 camera_ids 列表
    camera_ids = [f.split('.')[0] for f in os.listdir(DETECTED_FRAMES_DIR) if f.endswith('.png')]
    camera_ids.sort()
    return render_template('index.html', camera_ids=camera_ids)

@app.route('/image/<camera_id>')
def image(camera_id):
    # 指定图像文件路径
    image_path = os.path.join(DETECTED_FRAMES_DIR, f'{camera_id}.png')
    # 使用 Flask 的 send_from_directory 发送文件，移除了不支持的 cache_timeout 参数
    response = make_response(send_from_directory(os.path.dirname(image_path), os.path.basename(image_path)))
    # 设置 HTTP 头来禁用缓存，确保图像实时更新
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/camera/<camera_id>')
def camera(camera_id):
    return render_template('camera.html', camera_id=camera_id)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)  # Flask app 运行在 localhost 的 8000 端口

