from flask import Flask,render_template,request, jsonify,send_file

import os
import subprocess
from flask import Response
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

SCRIPT_PATH = '/home/prashant/Documents/legion/personal_projects/soccer/football_heatmap/heatmap_main.py'  # Path to your Python inference script

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400

    video = request.files['video']

    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded video
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

    return jsonify({'message': f'File uploaded successfully: {video.filename}', 'video_name': video.filename})


def convert_to_mp4(input_path, output_path):
    command = [
        'ffmpeg', '-y','-i', input_path, '-c:v', 'libx264', '-c:a', 'aac', output_path
    ]
    subprocess.run(command, check=True)


@app.route('/run_inference', methods=['POST'])
def run_inference():
    video_name = request.json.get('video_name')
    if not video_name:
        return jsonify({'error': 'No video specified for inference'}), 400

    input_video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
    
    if not os.path.exists(input_video_path):
        return jsonify({'error': 'Video file not found'}), 404
   

    
    
    video_name_final,ext= os.path.splitext(video_name)
    output_video_name = f'combined_output_{video_name_final}.avi'
    output_video_path_script=os.path.join(app.config['OUTPUT_FOLDER'],output_video_name)
    print("Output video path is: ",output_video_path_script)
   
    normal_detection = True  # or False
    try:
        # Run the inference script using subprocess
        command = [
            'python', SCRIPT_PATH,  # Path to your inference script
            '--video_path', input_video_path,
            '--output_video', app.config['OUTPUT_FOLDER']#output_video_path_script
        ]
        if normal_detection:
            command.append('--normal_detection')
        subprocess.run(command, check=True)
        output_video_mp4= os.path.join(app.config['OUTPUT_FOLDER'],f'combined_output_{video_name_final}.mp4')
       
        print("The output video name is:------------ ",output_video_name)
        convert_to_mp4(output_video_path_script,output_video_mp4)
       
        return jsonify({'message': 'Inference completed', 'output_video': f'combined_output_{video_name_final}.mp4'})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Inference script failed with error: {e}'}), 500


@app.route('/download/<filename>')
def download_video(filename):
    print("The filename that is being returned is---------",filename)
    output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    print("full output video path is_______+++: ",output_video_path)
    if not os.path.exists(output_video_path):
        return jsonify({'error': 'Output video not found'}), 404

    return Response(open(output_video_path, "rb"), mimetype="'video/mp4")


@app.route('/check_heatmap', methods=['POST'])
def check_heatmap():
    video_to_image = request.json.get('image_name')
    image_heatmap,ext= os.path.splitext(video_to_image)
    image_heatmap_name=image_heatmap+"_heatmap"+'.png'
    heatmap_image_path = os.path.join(app.config['OUTPUT_FOLDER'], image_heatmap_name)
    print("The heatmap image path is: ",heatmap_image_path)
    if not os.path.exists(heatmap_image_path):
        return jsonify({'error': f'No heatmap image fodun:'}), 500
    return jsonify({'message': 'Heatmap loaded', 'output_image': f'{image_heatmap_name}'})
   
        


@app.route('/download_heatmap/<filename>')
def download_heatmap(filename):
    image_name=filename
    output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], image_name)
    print("full output image path is______ ",output_image_path)
    return send_file(output_image_path, mimetype='image/jpeg')
if __name__=="__main__":
    app.run(debug=True)