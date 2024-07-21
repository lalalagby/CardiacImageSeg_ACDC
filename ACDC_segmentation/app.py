from flask import Flask, request, jsonify, send_file, url_for
import os
from werkzeug.utils import secure_filename
from ml_module import process_image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/segmented'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


@app.route('/')
def index():
    return send_file('templates/index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 处理图片
    processed_image_path = process_image(filepath,app.config['PROCESSED_FOLDER'])
    processed_image_url = url_for('static', filename=f'segmented/{os.path.basename(processed_image_path)}')

    return jsonify({'success': True, 'imageUrl': processed_image_url})


if __name__ == '__main__':
    app.run(debug=True)
