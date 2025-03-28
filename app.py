from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
from ultralytics import YOLO

# Initialize the Flask app
app = Flask(__name__)

# Configure upload and predictions directories
UPLOAD_FOLDER = 'uploads'
PREDICTIONS_FOLDER = 'predictions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER

# Load the YOLO model (adjust the path if necessary)
model = YOLO('detect/train/weights/best.pt')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part", 400
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Run prediction on the saved image
            results = model.predict(source=file_path, imgsz=640, conf=0.3, iou=0.5)
            detections = results[0]
            total_fish = len(detections.boxes)
            
            # Annotate the image (YOLOv8 provides a .plot() method to draw boxes on the image)
            annotated_image = detections.plot()  # returns a numpy array with drawn predictions

            # Save the annotated image to the predictions folder
            output_image_path = os.path.join(app.config['PREDICTIONS_FOLDER'], file.filename)
            cv2.imwrite(output_image_path, annotated_image)

            # Render the result template with the count and file name
            return render_template('result.html', total_fish=total_fish, predicted_image=file.filename)
    return render_template('index.html')

# Routes to serve the uploaded and predicted images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predictions/<filename>')
def predicted_file(filename):
    return send_from_directory(app.config['PREDICTIONS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
