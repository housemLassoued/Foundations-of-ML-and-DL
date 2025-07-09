
<p align="center">
  <em>This repository contains a collection of basic machine learning and deep learning notebooks, providing a starting point for learning these technologies.</em>
</p>

<hr/>
<h1 align="center">🩺 Breast Cancer Detection Project</h1>

<h2>💡 Project Overview</h2>

<p>
  Automatic detection of breast tumors from medical imaging using deep learning.
</p>

<h2>🛠️ Methodology</h2>

<p>
  The model leverages <strong>transfer learning</strong> with <strong>InceptionV3</strong> pre-trained on ImageNet, combined with a multilayer perceptron (MLP) as the classifier.
</p>

<h2>⚙️ Training Process</h2>

<ul>
  <li><strong>Phase 1:</strong>
    <ul>
      <li>All layers of InceptionV3 unfrozen</li>
      <li>Optimizer: Adam (learning rate = 1e-5)</li>
      <li>Loss: Binary Cross Entropy</li>
      <li>50 epochs</li>
    </ul>
  </li>
  <li><strong>Phase 2:</strong>
    <ul>
      <li>Only the last 20 layers unfrozen</li>
      <li>Same optimizer and loss function</li>
      <li>Additional 50 epochs</li>
    </ul>
  </li>
</ul>

<h2>⏳ Callbacks Used</h2>

<ul>
  <li><strong>ModelCheckpoint:</strong> Save the best-performing model</li>
  <li><strong>EarlyStopping:</strong> Stop training early if no improvement after 5 epochs</li>
  <li><strong>ReduceLROnPlateau:</strong> Reduce learning rate if loss plateaus for 3 epochs</li>
</ul>

<h2>📈 Evaluation</h2>

<p>
  The model was evaluated on a test set using the following metrics:
</p>

<ul>
  <li>Accuracy</li>
  <li>Precision</li>
  <li>Recall</li>
  <li>F1 Score</li>
</ul>

<h2>🎯 Objective</h2>

<p>
  This model is designed to be saved and deployed in clinical environments to support early detection of breast cancer.
</p>

<h2>🧰 Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>TensorFlow</li>
  <li>Keras</li>
</ul>

<hr/>

<p align="center">
  <img src="https://media.giphy.com/media/L1R1tvI9svkIWwpVYr/giphy.gif" width="300" alt="Coding Motivation"/>
</p>

<h1 align="center">🏃‍♂️ Calories Burnt Prediction Project</h1>

<p align="center">
  <em>This project aims to automatically predict caloric expenditure based on physiological and activity-related features.</em>
</p>

<hr/>

<h2>🎯 Objective</h2>

<p>
  Predict daily energy expenditure to help individuals monitor and optimize their health and fitness.
</p>

<h2>📊 Data Overview</h2>

<ul>
  <li><strong>Type:</strong> Tabular data</li>
  <li><strong>Source:</strong> Excel file containing the following variables:
    <ul>
      <li>Gender</li>
      <li>Age</li>
      <li>Height</li>
      <li>Weight</li>
      <li>Duration</li>
      <li>Heart Rate</li>
      <li>Body Temperature</li>
    </ul>
  </li>
</ul>

<h2>🛠️ Methodology</h2>

<ul>
  <li><strong>Data Exploration & Analysis</strong>
    <ul>
      <li>Studied correlations between features and the target variable</li>
      <li>Removed less informative features to reduce complexity</li>
    </ul>
  </li>
  <li><strong>Preprocessing</strong>
    <ul>
      <li>Encoding of categorical variables</li>
      <li>Outlier detection and treatment</li>
    </ul>
  </li>
  <li><strong>Data Splitting</strong>
    <ul>
      <li>Training set</li>
      <li>Validation set</li>
    </ul>
  </li>
</ul>

<h2>🤖 Model Training</h2>

<ul>
  <li>Linear Regression</li>
  <li>Random Forest Regressor</li>
  <li>XGBoost Regressor</li>
</ul>

<h2>📈 Evaluation</h2>

<ul>
  <li><strong>Loss Function:</strong> Mean Absolute Error (MAE)</li>
</ul>

<h2>🚀 Deployment</h2>

<p>
  The best-performing model is saved and ready to be integrated into an application to predict daily energy expenditure.
</p>

<h2>🧰 Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>scikit-learn</li>
  <li>Pandas</li>
</ul>

<hr/>

<p align="center">
  <img src="https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif" width="300" alt="Growth Motivation"/>
</p>
 
<h1 align="center">☔ Classifying Rainy Days Project</h1>

<p align="center">
  <em>This project predicts whether a day will be rainy based on tabular meteorological data.</em>
</p>

<hr/>

<h2>🎯 Objective</h2>

<p>
  Predict rainfall occurrence to support decision-making in agriculture and weather forecasting.
</p>

<h2>📊 Variables Used</h2>

<ul>
  <li>Pressure</li>
  <li>Temperature</li>
  <li>Humidity</li>
  <li>Cloud Cover</li>
  <li>Sunshine</li>
  <li>Wind Direction</li>
  <li>Wind Speed</li>
</ul>

<h2>🔄 Data Processing Pipeline</h2>

<ul>
  <li><strong>Data Analysis</strong>
    <ul>
      <li>Studied correlations with the target variable (rainfall)</li>
      <li>Removed low-impact features</li>
    </ul>
  </li>
  <li><strong>Preprocessing</strong>
    <ul>
      <li>Encoding categorical variables</li>
      <li>Handling outliers and missing values</li>
      <li>Normalizing features (mean = 0, std = 1)</li>
    </ul>
  </li>
  <li><strong>Class Imbalance Handling</strong>
    <ul>
      <li>Upsampling the minority class (no rain) to balance the dataset</li>
    </ul>
  </li>
</ul>

<h2>⚙️ Model Training & Validation</h2>

<ul>
  <li>Split data into training and validation sets</li>
  <li><strong>Models Tested:</strong>
    <ul>
      <li>Random Forest Classifier</li>
      <li>Support Vector Classifier (SVC)</li>
      <li>XGBoost Classifier</li>
      <li>Decision Tree Classifier</li>
    </ul>
  </li>
</ul>

<h2>🏆 Best Model Results</h2>

<ul>
  <li><strong>Selected Model:</strong> Random Forest Classifier</li>
  <li><strong>Accuracy:</strong> 87%</li>
  <li><strong>Precision:</strong> 91%</li>
  <li><strong>Recall:</strong> 82%</li>
  <li><strong>F1 Score:</strong> 86%</li>
</ul>

<h2>🚀 Deployment</h2>

<p>
  The final model is ready to be integrated into an application to predict rainfall and assist agricultural and meteorological planning.
</p>

<h2>🧰 Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>scikit-learn</li>
  <li>Pandas</li>
</ul>

<hr/>

<p align="center">
  <img src="https://media.giphy.com/media/juua9i2c2fA0AIp2iq/giphy.gif" width="300" alt="AI Developer Coding"/>
</p>


<h1 align="center">💳 Credit Card Fraud Detection Project</h1>

<p align="center">
  <em>Detecting fraudulent credit card transactions using supervised classification to enhance transaction security.</em>
</p>

<hr/>

<h2>🎯 Objective</h2>

<p>
  Build a predictive model to identify fraudulent transactions and strengthen banking security systems.
</p>

<h2>📊 Data Overview</h2>

<ul>
  <li><strong>Type:</strong> Tabular data (Excel file)</li>
  <li><strong>Content:</strong> Numeric variables only</li>
  <li><strong>Class Imbalance:</strong>
    <ul>
      <li>Normal Transactions: 284,315</li>
      <li>Fraudulent Transactions: 492</li>
    </ul>
  </li>
</ul>

<h2>🔄 Processing Pipeline</h2>

<ul>
  <li><strong>Imbalance Handling</strong>
    <ul>
      <li>Upsampling the minority (fraudulent) class to balance the dataset</li>
    </ul>
  </li>
  <li><strong>Data Splitting</strong>
    <ul>
      <li>Created training and validation sets</li>
    </ul>
  </li>
</ul>

<h2>⚙️ Model Training</h2>

<ul>
  <li><strong>Tested Models:</strong>
    <ul>
      <li>Random Forest Classifier</li>
      <li>Support Vector Classifier (SVC)</li>
      <li>Gaussian Naive Bayes</li>
      <li>Decision Tree Classifier</li>
    </ul>
  </li>
</ul>

<h2>🏆 Best Model Results</h2>

<ul>
  <li><strong>Selected Model:</strong> Random Forest Classifier</li>
  <li><strong>Accuracy:</strong> 99%</li>
  <li><strong>Precision:</strong> 91%</li>
  <li><strong>Recall:</strong> 79%</li>
  <li><strong>F1 Score:</strong> 85%</li>
</ul>

<h2>🚀 Deployment</h2>

<p>
  The final model is ready to be deployed in transaction monitoring applications to detect and prevent credit card fraud in real time.
</p>

<h2>🧰 Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>scikit-learn</li>
  <li>Pandas</li>
  <li>NumPy</li>
</ul>

<hr/>

<p align="center">
  <img src="https://media.giphy.com/media/QpVUMRUJGokfqXyfa1/giphy.gif" width="320" alt="Deep Learning"/>
</p>
    


<h1 align="center">✉️ Spam Email Detection</h1>

<p align="center">
  <em>Automatically detecting unwanted emails to enhance inbox organization and security.</em>
</p>

<hr/>

<h2>🎯 Objective</h2>

<p>
  Build a classification model to detect spam emails and streamline inbox management.
</p>

<h2>📊 Dataset Overview</h2>

<ul>
  <li><strong>Type:</strong> Tabular data (Excel file)</li>
  <li><strong>Variables:</strong>
    <ul>
      <li><code>Message</code>: Email text (in English)</li>
      <li><code>Category</code>: Label (<code>ham</code> or <code>spam</code>)</li>
    </ul>
  </li>
  <li><strong>Class Distribution:</strong>
    <ul>
      <li>Ham: 4,825 messages</li>
      <li>Spam: 747 messages</li>
    </ul>
  </li>
</ul>

<h2>🧪 Processing Pipeline</h2>

<ul>
  <li><strong>Label Encoding</strong>
    <ul>
      <li><code>ham → 0</code></li>
      <li><code>spam → 1</code></li>
    </ul>
  </li>
  <li><strong>Class Rebalancing</strong>
    <ul>
      <li>Upsampled spam messages to fix class imbalance</li>
    </ul>
  </li>
  <li><strong>Text Cleaning</strong>
    <ul>
      <li>Lowercasing</li>
      <li>Removing non-alphabetic characters</li>
      <li>Stop words removal (English)</li>
      <li>Word lemmatization</li>
    </ul>
  </li>
  <li><strong>Vectorization</strong>
    <ul>
      <li>TF-IDF used to extract weighted features from text</li>
    </ul>
  </li>
</ul>

<h2>🧠 Model Training & Evaluation</h2>

<ul>
  <li><strong>Models Tested:</strong>
    <ul>
      <li>Logistic Regression</li>
      <li>Random Forest Classifier</li>
    </ul>
  </li>
  <li><strong>Best Model:</strong> <code>RandomForestClassifier</code></li>
  <li><strong>Accuracy:</strong> <code>99%</code></li>
</ul>

<h2>🚀 Deployment</h2>

<p>
  The final model is ready to be integrated into a messaging application for automatic spam filtering and improved user experience.
</p>

<h2>🧰 Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>scikit-learn</li>
  <li>Pandas</li>
  <li>NLTK / spaCy (for NLP preprocessing)</li>
</ul>

<hr/>

<p align="center">
  <img src="https://media.giphy.com/media/coxQHKASG60HrHtvkt/giphy.gif" width="320" alt="Neural Networks"/>
</p>
            
       
<h1 align="center">🐕 Dog Breed Classification</h1>

<p align="center">
  <em>Automatically classifying dog images into 70 different breeds using a deep learning approach.</em>
</p>

<hr/>

<h2>🎯 Objective</h2>

<p>
  Develop a model to predict the breed of a dog based on an input image.
</p>

<h2>📊 Dataset Overview</h2>

<ul>
  <li><strong>Source:</strong> Excel file containing image paths and labels</li>
  <li><strong>Training Set:</strong> 7,946 RGB images (224×224)</li>
  <li><strong>Test/Validation Set:</strong> 700 images</li>
  <li><strong>Number of Classes:</strong> 70 breeds</li>
</ul>

<h2>🧪 Processing Pipeline</h2>

<ul>
  <li><strong>Preprocessing</strong>
    <ul>
      <li>Image normalization</li>
      <li>Data augmentation:
        <ul>
          <li>Zoom</li>
          <li>Horizontal flip</li>
        </ul>
      </li>
    </ul>
  </li>
  <li><strong>Modeling</strong>
    <ul>
      <li><code>ResNet101V2</code> pretrained on ImageNet as feature extractor (frozen weights)</li>
      <li>Multilayer Perceptron (MLP) head for multiclass classification</li>
    </ul>
  </li>
</ul>

<h2>🧠 Training</h2>

<ul>
  <li><strong>Optimizer:</strong> Adam</li>
  <li><strong>Epochs:</strong> 25</li>
  <li><strong>EarlyStopping:</strong> Patience = 10 epochs</li>
</ul>

<h2>✅ Results</h2>

<ul>
  <li><strong>Accuracy:</strong> 93%</li>
</ul>

<h2>🚀 Deployment</h2>

<p>
  The model is ready for deployment in an application that automatically predicts dog breeds from images.
</p>

<h2>🧰 Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>TensorFlow</li>
  <li>Keras</li>
</ul>

<hr/>

<p align="center">
  <img src="https://media.giphy.com/media/qgQUggAC3Pfv687qPC/giphy.gif" width="400" alt="Coding GIF"/>
</p>

<h1 align="center">👁️✋ Face and Hand Landmarks Detection</h1>

<p align="center">
  <em>Real-time detection of face, hand, and body landmarks using MediaPipe and OpenCV.</em>
</p>

<hr/>

<h2>🎯 Project Objective</h2>

<p>
  This project leverages the <strong>MediaPipe</strong> library in combination with <strong>OpenCV</strong> to detect and display landmarks on the face, hands, and body of a person captured in real time by a webcam.
  Landmarks are used to identify specific features such as the positions of eyes, nose, fingers, etc.
  This type of system can be applied to various use cases:
</p>

<ul>
  <li>🔒 <strong>Security:</strong> Detecting and identifying individuals in public places (e.g., shopping centers)</li>
  <li>🖐️ <strong>Human-Computer Interaction:</strong> Controlling systems via gestures</li>
  <li>👥 <strong>Behavior Analysis:</strong> Monitoring and analyzing human behavior</li>
</ul>

<h2>🛠️ Libraries Used</h2>

<ul>
  <li><strong>OpenCV (cv2):</strong> Capturing and processing webcam images</li>
  <li><strong>MediaPipe (mp):</strong> Detecting face, hand, and body landmarks</li>
  <li><strong>time:</strong> Calculating frames per second (FPS)</li>
</ul>

<h2>⚙️ Processing Steps</h2>

<h3>1️⃣ Video Capture Initialization</h3>

<ul>
  <li>Video capture is initialized using <code>cv2.VideoCapture(0)</code>, where 0 is the default camera.</li>
  <li>Frames are resized to <strong>800×600</strong> for better readability and faster processing.</li>
</ul>

<h3>2️⃣ Loading the Holistic Model</h3>

<ul>
  <li>The pre-trained <strong>MediaPipe Holistic</strong> model is loaded with:
    <ul>
      <li><code>min_detection_confidence=0.5</code></li>
      <li><code>min_tracking_confidence=0.5</code></li>
    </ul>
  </li>
  <li>This model can simultaneously detect:
    <ul>
      <li>Face landmarks (eyes, nose, mouth, etc.)</li>
      <li>Hand landmarks (left and right)</li>
      <li>Body pose landmarks</li>
    </ul>
  </li>
</ul>

<h3>3️⃣ Frame Processing</h3>

For each frame captured:

<ul>
  <li><strong>Color conversion:</strong> Convert BGR (OpenCV) to RGB (MediaPipe)</li>
  <li><strong>Disable writeable flag:</strong> Optimize performance during detection</li>
  <li><strong>Landmark detection:</strong> Process the frame with Holistic model</li>
  <li><strong>Enable writeable flag:</strong> Allow annotations</li>
  <li><strong>Convert RGB back to BGR:</strong> For OpenCV display</li>
</ul>

<h3>4️⃣ Drawing Landmarks</h3>

<ul>
  <li>Face landmarks: Drawn with colored contours</li>
  <li>Hand landmarks: Drawn with connections for left and right hands</li>
  <li>Body landmarks: Optionally drawn for full body pose</li>
</ul>

<h3>5️⃣ FPS Calculation</h3>

<ul>
  <li>Elapsed time between frames is measured</li>
  <li>FPS is calculated as the inverse of elapsed time</li>
  <li>Displayed at the top of the video feed for performance monitoring</li>
</ul>

<h3>6️⃣ Display and Exit</h3>

<ul>
  <li>The final frame with landmarks is shown in a window titled <strong>"Facial and Hand Landmarks"</strong></li>
  <li>The loop continues until the user presses <code>'q'</code> to stop and close the window</li>
</ul>

<h2>🌟 Potential Applications</h2>

<ul>
  <li>🔐 <strong>Security:</strong> Identifying individuals using facial landmarks</li>
  <li>✋ <strong>Gesture Interaction:</strong> Controlling applications with hand or body gestures</li>
  <li>🏃‍♂️ <strong>Sports Analysis:</strong> Tracking posture and movements to improve performance</li>
  <li>🎮 <strong>Interactive Games:</strong> Controlling virtual objects via body and hand movements</li>
</ul>

<hr/>

<h2>🧰 Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>MediaPipe</li>
  <li>OpenCV</li>
</ul>
<p align="center">
  <img src="https://media.giphy.com/media/26tn33aiTi1jkl6H6/giphy.gif" width="400" alt="Matrix GIF"/>
</p>

<h1 align="center">🩻 Lumbar Coordinate Classification</h1>

<p align="center">
  <em>Multiclass classification of lumbar spine degenerative pathologies for the RSNA 2024 Competition</em>
</p>

<hr/>

<h2>🎯 Objective</h2>

<p>
  The goal of this project is to classify medical images into four classes—LSD, OSF, SPIDER, and TSEG—in the context of the RSNA 2024 Lumbar Spine Degenerative Classification competition. This project aims to address a specific problem related to analyzing degenerative pathologies of the lumbar spine. Each class represents a distinct category or task for characterizing and diagnosing anomalies on medical scans:
</p>

<ul>
  <li><strong>LSD</strong>: Pathology classification</li>
  <li><strong>OSF</strong>: Structural orientation analysis</li>
  <li><strong>SPIDER</strong>: Detection of complex anomalies</li>
  <li><strong>TSEG</strong>: Anatomical structure segmentation</li>
</ul>

<p>
  This contributes to more precise and automated medical diagnoses.
</p>

<h2>📊 Dataset Overview</h2>

<ul>
  <li><strong>Total Images:</strong> 1,237</li>
  <li><strong>Class Distribution:</strong>
    <ul>
      <li>LSD: 515 images</li>
      <li>OSF: 34 images</li>
      <li>SPIDER: 210 images</li>
      <li>TSEG: 478 images</li>
    </ul>
  </li>
</ul>

<p>
  The strong class imbalance can cause bias during training, where the model may learn to predict majority classes better. Therefore, balancing the dataset was necessary to ensure equal representation for all classes.
</p>

<h2>⚙️ Processing Pipeline</h2>

<ul>
  <li><strong>Dataset Splitting:</strong>
    <ul>
      <li>60% training</li>
      <li>20% validation</li>
      <li>20% testing</li>
    </ul>
  </li>
  <li><strong>Image Preprocessing:</strong>
    <ul>
      <li>Normalization of pixel intensities to [0,1]</li>
      <li>Resizing from 256×256 to 224×224</li>
      <li>Conversion from BGR to RGB</li>
      <li>Ensuring 3 channels per image</li>
    </ul>
  </li>
  <li><strong>Data Augmentation (Training Set):</strong>
    <ul>
      <li>Random rotations (±10 degrees)</li>
      <li>Width and height shifts (10%)</li>
      <li>Zoom transformations</li>
    </ul>
  </li>
</ul>

<h2>🧠 Model Architecture</h2>

<ul>
  <li><strong>Feature Extractor:</strong> Pretrained ResNet50 (frozen weights)</li>
  <li><strong>Classification Head:</strong> Multilayer Perceptron (MLP) with softmax activation</li>
</ul>

<h2>🛠️ Training Configuration</h2>

<ul>
  <li><strong>Epochs:</strong> 30 (Early stopping reduced to 17 epochs)</li>
  <li><strong>Optimizer:</strong> Adam</li>
  <li><strong>Learning Rate:</strong> 0.001</li>
  <li><strong>Loss Function:</strong> Sparse Categorical Crossentropy (suitable for multiclass classification)</li>
  <li><strong>Callbacks:</strong>
    <ul>
      <li>EarlyStopping (patience=10)</li>
    </ul>
  </li>
</ul>

<h2>✅ Results</h2>

<ul>
  <li><strong>Test Accuracy:</strong> 95.97%</li>
</ul>

<h2>🚀 Deployment</h2>

<p>
  The model is ready to be integrated into applications for automatic diagnosis and classification of lumbar spine degenerative conditions, supporting more efficient and accurate medical decision-making.
</p>

<h2>🧰 Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>TensorFlow</li>
  <li>Keras</li>
</ul>

<hr/>

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcm55eDJhZ2h3YzVpZTBqbjBlbHR6bm9jOHR1YW5qZ3J6Mm9yMzd2aCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/WFZvB7VIXBgiz3oDXE/giphy.gif" width="300" alt="Medical AI"/>
</p>

