# Fish Counting App - Underwater Object Detection with YOLOv8

A professional web application built with Flask and YOLOv8 for detecting and counting fish in underwater images. This project uses a custom-trained YOLOv8 model to accurately identify and count 13 species of fish. The model is designed to support underwater monitoring, marine research, and prototype development in computer vision.

## Features

- **Advanced Detection:** Utilizes YOLOv8 for fast and accurate detection of underwater species.
- **Multi-species Support:** Trained on 13 fish species:
  - Acanthuridae - Surgeonfishes
  - Balistidae - Triggerfishes
  - Carangidae - Jacks
  - Ephippidae - Spadefishes
  - Labridae - Wrasse
  - Lutjanidae - Snappers
  - Pomacanthidae - Angelfishes
  - Pomacentridae - Damselfishes
  - Scaridae - Parrotfishes
  - Scombridae - Tunas
  - Serranidae - Groupers
  - Shark - Selachimorpha
  - Zanclidae - Moorish Idol
- **User-Friendly Interface:** Built with Flask and Bootstrap featuring an ocean-inspired theme.
- **Easy Image Upload & Result Display:** Upload images in common formats (JPEG, PNG, etc.) and view annotated predictions along with fish counts.



## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Pip](https://pip.pypa.io/)
- [Virtualenv](https://virtualenv.pypa.io/) (recommended)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Muhammad-waqas1/fish-counting-app.git
   cd fish-counting-app

2. **Create and activate a virtual environment:**

   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate


3. **Install the required dependencies:**

   ```bash
    pip install -r requirements.txt

_Note: Ensure your requirements.txt includes packages such as Flask, ultralytics, opencv-python, etc._

4. **Place your trained model weights:**

Ensure the trained model file (*best.pt*) is located in *detect/train/weights/*.
If you haven’t trained the model yet, follow the training instructions below.

## Training the Model
If you wish to train the model from scratch:

1. Prepare your dataset:
Create a *data.yaml* file containing paths to your training, validation, and test images along with class information.

2. **Run the training script:**

   ```bash
    python main.py

This will retrain the YOLOv8 model for the specified number of epochs. Adjust the script as needed for your dataset paths and parameters.


## Running the Web Application

1. **Run the Flask app:**

    ```bash
        python main.py

2. **Access the application:**

Open your web browser and navigate to `http://127.0.0.1:5000/` to access the site.

3. **Usage:**

- **Home Page**: Upload an underwater image.

- **Prediction:** View the annotated image along with the detected fish count.

- **About:** Learn more about the model, dataset, and its applications.

- **Contact:** Get in touch or connect via GitHub/Kaggle.

## About the Model

This project leverages the YOLOv8 architecture for underwater object detection. The model is trained to detect 13 fish species, making it suitable for various applications including:

- Underwater object detection systems

- Marine research and monitoring

- Prototyping advanced computer vision systems for aquatic environments


## Meet the Team

- **Muhammad Waqas**  
  Sole creator and developer of this project. Passionate about computer vision and marine biology.  
  Connect on: [GitHub](https://github.com/Muhammad-waqas1) | [Kaggle](https://www.kaggle.com/waqas010)

## License

This project is open source and available under the [MIT License](LICENSE).

## Issues

If you encounter any issues or have suggestions for improvements, please feel free to [open an issue](https://github.com/Muhammad-waqas1/fish-counting-app/issues) on GitHub.


----

### Additional Notes

- **requirements.txt:**  
  Make sure to create a `requirements.txt` file listing all the dependencies (e.g., Flask, ultralytics, opencv-python).

- **.gitignore:**  
  Your `.gitignore` should exclude virtual environments, runtime folders (uploads, predictions, runs), model weights (`*.pt`), and any OS or IDE-specific files.

- **Customization:**  
  You can further adjust the content and styling of the README and other documentation as needed.


  ---

_If you enjoyed this project or found it useful, please consider giving it a ⭐ on GitHub!_
