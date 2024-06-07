from flask import Flask, render_template, request, redirect, url_for
import os
from utils import video_processing

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files.get('file')
    if file:
        filename = "video.mp4"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        processed_filename = "processed_" + filename
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        
        video_processing.process_video(filepath, processed_filepath)
        
        return redirect(url_for('result'))
    
    return redirect(request.url)

@app.route('/process/<filename>')
def process_video(filename):
    return render_template('play.html', filename=filename)

@app.route('/result')
def result():
    processed_video_path = 'uploads/processed_video.mp4'
    original_video_path = 'uploads/video.mp4'
    analysis_file_path = 'uploads/processed_video.mp4.txt'
    return render_template('result.html', processed_video=processed_video_path, original_video=original_video_path, analysis_file=analysis_file_path)

if __name__ == '__main__':
    app.run(debug=True)
