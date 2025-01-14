# Face Recognition



"Face Recognition is a powerful and user-friendly Python library built on dlib’s state-of-the-art deep learning models, enabling accurate face detection, recognition, and feature manipulation with an impressive 99.38% accuracy on the Labeled Faces in the Wild benchmark."







## Acknowledgements

Acknowledgements

dlib Library – For providing the foundational machine learning models and facial recognition algorithms.

OpenCV – For enabling real-time computer vision capabilities in Python.

NumPy – For efficient numerical operations and data handling.

Scikit-learn – For supporting machine learning techniques used in face comparison.

Python Software Foundation – For maintaining the Python language that powers this project.

Docker – For simplifying deployment across different platforms.

Community Contributors – For continuous improvements, bug fixes, and feature suggestions.






## API Reference

#### Get all items

```http
GET /api/face_recognition
```


| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_key` | `string` | **Required**. Your API key |
| `image` | `file` | **Required**. The image file to perform face detection on. |
| `model` | `string` | Optional. Detection model: hog (default) or cnn
 |
| `tolerance` | `string` | Optional. Face comparison tolerance (default 0.6)
|





## Appendix

Any additional information goes here

Face Detection Models
        
        HOG (Histogram of Oriented Gradients)

                 Default model for face detection.
                 Faster and suitable for CPU-based systems.
                 Best for real-time applications with limited computational resources.
CNN (Convolutional Neural Network):

                More accurate but computationally intensive.
                Requires GPU acceleration (NVIDIA CUDA) for optimal performance.
                Recommended for high-precision face detection tasks.

Face Encoding and Comparison

                Face recognition works by converting facial features into a 128-dimensional encoding.
                The comparison is done by calculating the Euclidean distance between face encodings.
                        Tolerance Setting:
                                  Default is 0.6.
                                  Lower values make the comparison stricter, reducing false positives.
                                  Higher values allow more flexibility but may increase false matches.


Limitations

                  Children's Faces: Lower accuracy due to limited training data on children.
                  Ethnic Diversity: Performance may vary across different demographic groups.
                  Lighting and Angles: Poor lighting and extreme angles can affect recognition accuracy.

Best Practices

          Use high-quality, well-lit images for better recognition results.
          Preprocess images (resize, normalize) to improve model performance.
          For real-time detection, use the cnn model with GPU support.

Security Considerations

         Ensure secure storage and handling of face data to protect user privacy.
         Use encrypted connections (HTTPS) for API communication.
Implement proper access controls for sensitive data handling.

Further Enhancements

     Integrate with OpenCV for live video face recognition.
     Use clustering algorithms for organizing large photo datasets.
     Deploy with Docker for scalable and consistent environments.
     
This appendix provides additional insights into optimizing and understanding the face recognition system for various applications.
## Authors

- [@KUSHBU](https://github.com/Kushbhukumari)


##  Badges



[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)



## Contributing

How to Contribute

Fork the repository
Clone your forked repo:
bash

Copy code:
git clone https://github.com/your-username/face_recognition.git

Create a new branch for your feature or bug fix:
bash

Copy code:
git checkout -b feature-name

Commit your changes:
bash

Copy code:
git push origin feature-name




## Demo

Insert gif or link to demo

https://github.com/Kushbhukumari/FACE_OF_RECOGNITION_.git
## Deployment

For Node.js environments: If you're running a Node.js script for deployment, the following command could be used to deploy:

bash
Copy code

```bash
  npm run deploy
```

To deploy the Face Recognition project, follow these steps:

Prerequisites
Make sure you have all dependencies installed and configured properly before deployment.

Install required libraries and dependencies, for example, using pip:

```bash
pip install -r requirements.txt
```

Set up environment variables, such as your API keys or configuration details.

Deploying the Project
To deploy the project, you can run the following command (for example, if you are using Docker or a cloud service):

```bash
  docker-compose up --build
```

## Installation

Installation in PyCharm in Command prompt

    pip install tensorflow==2.10.0

    pip install opencv-python==4.5.5.64

    pip install numpy==1.21.6

run the code

## Installation: 1.Visit the Teachable Machine Website: Go to Teachable Machine

![image](https://github.com/user-attachments/assets/ef1ebfa4-2020-43fc-a12d-802e30662eba)

## 2.Create a New Project: Click on "Get Started" and select "Image Model" under the "New Project" section 3.Select Model Type: Choose the "Standard Image Model" option. image

![image](https://github.com/user-attachments/assets/7c9d2a31-e40a-4e17-a9b0-54628adfe6de)

## 3.Label Examples: Assign labels to each example image

![image](https://github.com/user-attachments/assets/8729ef01-315c-4470-87a2-b8982fe6a982)

## 4.Export the Model: Once training is complete, click on "Export the Model" and download the model files (a .zip file containing the model weights (.h5) and labels (.txt) files) image


## Implementation in Python 

1.Set Up Your Environment: Ensure you have Python 3.7 or higher installed.
2.Install Required Libraries: Install OpenCV and NumPy using pip: python ->pip install opencv-python numpy
3.Extract Model Files: Extract the downloaded .h5 and .txt files from the .zip archive and save them in your project directory.
4.Write Python Code: Use the following code to load the model and perform face recognition: from keras.models import load_model 


## CAMERA can be 0 or 1 based on default camera of your computer

     # Resize the raw image into (224-height,224-width) pixels
       image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

     # Show the image in a window
       cv2.imshow("Webcam Image", image)

     # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

     # Normalize the image array
        image = (image / 127.5) - 1

     # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

      # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

      # Listen to the keyboard for presses.
         keyboard_input = cv2.waitKey(1)

       # 27 is the ASCII for the esc key on your keyboard.
          if keyboard_input == 27:
                break

## CODE EXPLANATION

IMPORTS from keras.models import load_model: Imports the load_model function from Keras to load the pre-trained model. import cv2: Imports the OpenCV library for computer vision tasks. import numpy as np: Imports the NumPy library for numerical operations.

CONFIGURATION np.set_printoptions(suppress=True): Sets the NumPy print options to suppress scientific notation for clarity when printing.

LOAD MODELS AND LABELS model = load_model("keras_Model.h5", compile=False): Loads the pre-trained model from the file keras_Model.h5 without compiling it. class_names = open("labels.txt", "r").readlines(): Reads the class labels from the file labels.txt into a list.

CAMERA SETUP camera = cv2.VideoCapture(0): Opens the default camera (camera index 0) for capturing images.

MAIN LOOPS while True:: Starts an infinite loop to continuously capture images. ret, image = camera.read(): Captures an image from the camera. image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA): Resizes the image to 224x224 pixels to match the model's input size. cv2.imshow("Webcam Image", image): Displays the captured image in a window titled "Webcam Image".

PREPROCESS IMAGE image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3): Converts the image to a NumPy array and reshapes it to (1, 224, 224, 3) to match the model's input shape. image = (image / 127.5) - 1: Normalizes the image array to a range of [-1, 1]

MAKE PREDICTION prediction = model.predict(image): Uses the model to predict the class of the input image. index = np.argmax(prediction): Finds the index of the class with the highest confidence score. class_name = class_names[index]: Retrieves the class name corresponding to the predicted index. confidence_score = prediction[0][index]: Retrieves the confidence score of the predicted class.

DISPLAY RESULTS print("Class:", class_name[2:], end=""): Prints the predicted class name. print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%"): Prints the confidence score as a percentage.

HANDLE KEYBOARD INPUT keyboard_input = cv2.waitKey(1): Waits for keyboard input. if keyboard_input == 27:: Checks if the 'Esc' key (ASCII code 27) is pressed to break the loop.

RELEASE RESOURCE camera.release(): Releases the camera resource. cv2.destroyAllWindows(): Closes all OpenCV windows.

## Related

Here are some related projects

[Awesome README](https://github.com/ageitgey/face_recognition?tab=readme-ov-file)
## Screenshots
![Screenshot 2025-01-07 212613](https://github.com/user-attachments/assets/63ea50a9-6360-49cd-98d9-8f9d0c7c905e)
![Screenshot 2025-01-07 212633](https://github.com/user-attachments/assets/9862502f-de74-4491-a5fc-70e8f1e7c76a)

