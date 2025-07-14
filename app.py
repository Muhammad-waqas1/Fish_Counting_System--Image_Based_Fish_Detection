from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
from ultralytics import YOLO
from datetime import datetime
from collections import Counter

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
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Run YOLOv8 prediction
            results = model.predict(source=file_path, imgsz=640, conf=0.6, iou=0.5)
            detections = results[0]


            # Filter detections by confidence threshold
            conf_threshold = 0.6
            filtered_indices = [i for i, conf in enumerate(detections.boxes.conf) if conf > conf_threshold]

            # If no detections meet the threshold
            if not filtered_indices:
                total_fish = 0
                species_counts = {}
            else:
                filtered_boxes = detections.boxes[filtered_indices]
                class_ids = filtered_boxes.cls.cpu().numpy().astype(int)
                labels = model.names
                species_counts = dict(Counter([labels[cid] for cid in class_ids]))
                total_fish = len(filtered_boxes)

                # Update the detections object with filtered boxes (optional, for plotting)
                detections.boxes = filtered_boxes


            # Save annotated image
            annotated_image = detections.plot()
            output_path = os.path.join(app.config['PREDICTIONS_FOLDER'], filename)
            cv2.imwrite(output_path, annotated_image)

            # Timestamp
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            print("Species counts:", species_counts)
            print("Detections before filtering:", len(detections.boxes))
            print("Filtered detections:", total_fish)

            return render_template(
                'result.html',
                total_fish=total_fish,
                predicted_image=filename,
                original_image=filename,
                species_counts=species_counts,
                timestamp=timestamp
            )
            
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
