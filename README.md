
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
<hr/>
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

<h1 align="center">🎤 TED Talks Recommendation System</h1>

<p align="center">
  <em>Building an intelligent recommendation system to suggest TED talks based on content similarity, popularity, speaker, and topics.</em>
</p>

<hr/>

<h2>🎯 Objective</h2>

<p>
  The goal of this project is to develop a recommendation system for TED talks. Talks are suggested based on their content similarity, popularity, main speaker, tags, event, and themes.
</p>

<h2>📚 Data and Content</h2>

<ul>
  <li><strong>Data Source:</strong> TED talk dataset containing metadata and transcripts</li>
  <li><strong>Content Features:</strong>
    <ul>
      <li>Title</li>
      <li>Description</li>
      <li>Transcript</li>
    </ul>
  </li>
</ul>

<h2>⚙️ Processing Pipeline</h2>

<ul>
  <li><strong>Text Preprocessing:</strong>
    <ul>
      <li>Lowercasing all text</li>
      <li>Removing punctuation</li>
      <li>Removing stopwords</li>
      <li>Lemmatizing words to their base form to reduce noise</li>
    </ul>
  </li>
  <li><strong>Vectorization:</strong>
    <ul>
      <li>Using <code>TfidfVectorizer</code> from scikit-learn to transform text into numerical representations</li>
      <li>This method assigns low weights to frequent words and higher weights to less frequent but more meaningful words</li>
    </ul>
  </li>
  <li><strong>Similarity Measurement:</strong>
    <ul>
      <li>Calculating <code>cosine_similarity</code> between talk content vectors</li>
      <li>Identifying the top 5 most similar TED talks for recommendations</li>
    </ul>
  </li>
</ul>

<h2>💡 Recommendation Strategies</h2>

<ul>
  <li><strong>Content-Based Recommendations:</strong> Suggest talks similar in content to those the user liked</li>
  <li><strong>Popularity-Based Recommendations:</strong> Suggest the most viewed TED talks</li>
  <li><strong>Speaker-Based Recommendations:</strong> If a user engages with talks by a specific speaker, recommend other talks by the same main speaker</li>
  <li><strong>Tag-Based Recommendations:</strong> Recommend talks sharing common tags</li>
  <li><strong>Event-Based Recommendations:</strong> Suggest talks presented at the same event</li>
  <li><strong>Theme-Based Recommendations:</strong> For example, if a user prefers “funny” talks, the system recommends other humorous TED talks</li>
</ul>

<h2>🚀 Deployment</h2>

<p>
  This recommendation system follows principles commonly used in e-commerce websites and social media platforms to personalize content and enhance user engagement.
</p>

<h2>🧰 Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>scikit-learn</li>
  <li>Pandas</li>
  <li>NLP (Natural Language Processing)</li>
</ul>

<hr/>

<p align="center">
  <img src="https://media.giphy.com/media/f3iwJFOVOwuy7K6FFw/giphy.gif" width="400" alt="Developer Coding"/>
</p>

<h1 align="center">♻️ Recyclable and Household Waste Plastic Detection</h1>

<p align="center">
  <em>Multiclass classification of plastic waste items using deep learning on the Recyclable and Household Waste Classification dataset</em>
</p>

<hr/>

<h2>🎯 Objective</h2>

<p>
  The goal of this project is to develop a deep learning model capable of detecting plastic waste in household waste images and identifying the specific plastic item category. To achieve this, we leveraged the <a href="https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification">Recyclable and Household Waste Classification dataset</a> from Kaggle.
</p>

<p>
  This project aims to address the growing need for accurate sorting of recyclable materials by distinguishing between plastic and non-plastic objects and classifying plastic waste into detailed subcategories.
</p>

<h2>📊 Dataset Overview</h2>

<ul>
  <li><strong>Original Dataset Structure:</strong>
    <ul>
      <li>For each waste item, two folders: <code>default</code> and <code>real_world</code>, each containing 250 images</li>
    </ul>
  </li>
  <li><strong>Dataset Reorganization:</strong>
    <ul>
      <li>Identified plastic and non-plastic items</li>
      <li>Split images into:
        <ul>
          <li>75% training</li>
          <li>15% validation</li>
          <li>15% testing</li>
        </ul>
      </li>
      <li>New structure:
        <ul>
          <li><code>train/</code></li>
          <li><code>validation/</code></li>
          <li><code>test/</code></li>
        </ul>
        Each split contains:
        <ul>
          <li><code>plastic/</code>: all plastic items</li>
          <li><code>not_plastic/</code>: all non-plastic items</li>
        </ul>
      </li>
    </ul>
  </li>
</ul>

<p>
  For labeling, each image of a plastic item was assigned a distinct class label, while all non-plastic images shared a single label, resulting in a supervised multiclass classification problem.
</p>

<h2>⚙️ Processing Pipeline</h2>

<ul>
  <li><strong>Image Preprocessing:</strong>
    <ul>
      <li>Resized from 256×256×3 to 224×224×3</li>
      <li>Normalized pixel values to [0,1]</li>
    </ul>
  </li>
</ul>

<h2>🧠 Model Architecture</h2>

<ul>
  <li><strong>Feature Extractor:</strong> Pretrained VGG16 (frozen weights)</li>
  <li><strong>Classification Head:</strong> Multilayer Perceptron (MLP) with softmax activation for multiclass output</li>
</ul>

<h2>🛠️ Training Configuration</h2>

<ul>
  <li><strong>Epochs:</strong> 10</li>
  <li><strong>Optimizer:</strong> Adam</li>
  <li><strong>Learning Rate:</strong> 0.0001</li>
  <li><strong>Loss Function:</strong> Sparse Categorical Crossentropy</li>
  <li><strong>Frozen Weights:</strong> VGG16 base layers were not updated during training</li>
</ul>

<h2>✅ Results</h2>

<ul>
  <li><strong>Overall Test Accuracy:</strong> 91%</li>
</ul>

<p><strong>Classification Report:</strong></p>

<table align="center">
  <thead>
    <tr>
      <th>Class</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>plastic_cup_lids</td><td>0.81</td><td>0.82</td><td>0.82</td><td>73</td></tr>
    <tr><td>plastic_detergent_bottles</td><td>0.93</td><td>0.77</td><td>0.85</td><td>71</td></tr>
    <tr><td>plastic_food_containers</td><td>0.91</td><td>0.74</td><td>0.82</td><td>69</td></tr>
    <tr><td>plastic_shopping_bags</td><td>0.90</td><td>0.76</td><td>0.82</td><td>70</td></tr>
    <tr><td>plastic_soda_bottles</td><td>0.79</td><td>0.80</td><td>0.79</td><td>70</td></tr>
    <tr><td>plastic_straws</td><td>0.87</td><td>0.78</td><td>0.82</td><td>67</td></tr>
    <tr><td>plastic_trash_bags</td><td>0.96</td><td>0.71</td><td>0.82</td><td>69</td></tr>
    <tr><td>plastic_water_bottles</td><td>0.77</td><td>0.66</td><td>0.71</td><td>67</td></tr>
    <tr><td>disposable_plastic_cutlery</td><td>0.92</td><td>0.79</td><td>0.85</td><td>73</td></tr>
    <tr><td>Not Plastic</td><td>0.93</td><td>0.97</td><td>0.95</td><td>1471</td></tr>
  </tbody>
</table>

<h2>🚀 Deployment</h2>

<p>
  The model can be integrated into waste management and recycling systems to automate the identification and classification of plastic waste, promoting more efficient sorting and recycling processes.
</p>

<h2>🧰 Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>TensorFlow</li>
  <li>Keras</li>
</ul>

<hr/>

<p align="center">
  <img src="https://media.giphy.com/media/LMt9638dO8dftAjtco/giphy.gif" width="300" alt="Developer Coding"/>
</p>
<h1 align="center">🗣️ Simple Voice Assistant</h1>

<p align="center">
  <em>A Python-based voice assistant capable of interpreting and executing spoken commands in real time</em>
</p>

<hr/>

<h2>🎯 Objective</h2>

<p>
  This project aims to create an interactive voice assistant that interprets spoken commands through a microphone and performs specific actions accordingly. The assistant leverages multiple Python libraries to handle text-to-speech synthesis, speech recognition, external program execution, and access to online services such as Wikipedia, YouTube, and Gmail.
</p>

<h2>📚 Libraries Used</h2>

<ul>
  <li><strong>pyttsx3:</strong> Provides text-to-speech synthesis to convert text into spoken output.</li>
  <li><strong>speech_recognition:</strong> Captures and transcribes voice commands using Google Speech Recognition.</li>
  <li><strong>subprocess:</strong> Executes external programs (e.g., launching Chrome).</li>
  <li><strong>datetime:</strong> Retrieves the current time information.</li>
  <li><strong>wikipedia:</strong> Performs Wikipedia searches and returns short summaries.</li>
  <li><strong>webbrowser:</strong> Opens specific web pages in the default browser.</li>
  <li><strong>smtplib:</strong> Manages sending emails via SMTP.</li>
  <li><strong>pywhatkit:</strong> Plays YouTube videos or executes web actions.</li>
  <li><strong>pyjokes:</strong> Generates humorous jokes to entertain the user.</li>
</ul>

<h2>⚙️ Program Workflow</h2>

<ul>
  <li><strong>Voice Engine Initialization:</strong>
    <ul>
      <li>The voice engine (<code>pyttsx3</code>) is initialized to convert text to speech.</li>
      <li>Available voices are retrieved, and a specific voice (e.g., female) is selected for interaction.</li>
    </ul>
  </li>
  <li><strong>Speech Recognition:</strong>
    <ul>
      <li>The <code>cmd()</code> function manages speech recognition.</li>
      <li>Background noise is automatically adjusted to improve recognition accuracy.</li>
      <li>Audio is captured via the microphone and transcribed into text using Google Speech Recognition.</li>
      <li>If the command is not recognized or an error occurs, an appropriate message is displayed.</li>
    </ul>
  </li>
  <li><strong>Command Processing:</strong>
    <ul>
      <li>An infinite loop continuously listens for user commands.</li>
      <li>Each command is analyzed for specific keywords to trigger the corresponding action:</li>
      <ul>
        <li><strong>Greetings:</strong> If the command contains "bonjour", the assistant responds, "Bonjour! How can I help you?"</li>
        <li><strong>Launching Chrome:</strong> If "chrome" is mentioned, Chrome is opened via <code>subprocess</code>.</li>
        <li><strong>Wikipedia Search:</strong> If "wikipedia" is detected, the assistant extracts the search term, queries Wikipedia, and reads a three-sentence summary aloud.</li>
        <li><strong>Web Navigation:</strong> If "jumia" is mentioned, the Jumia website is opened in the default browser.</li>
        <li><strong>Time Announcement:</strong> If "time" is included, the current time is spoken.</li>
        <li><strong>YouTube Playback:</strong> If "play" is used, the assistant plays a YouTube video matching the extracted terms.</li>
        <li><strong>Email Sending:</strong> If "email" is mentioned, the assistant prompts for the message content and recipient address, then sends the email using <code>smtplib</code>.</li>
        <li><strong>Jokes:</strong> If "joke" is detected, a joke is read aloud.</li>
        <li><strong>Exit:</strong> If "au revoir" is spoken, the assistant says "Goodbye!" and exits the main loop.</li>
      </ul>
    </ul>
  </li>
  <li><strong>Error Handling:</strong>
    <ul>
      <li>If a command is unrecognized or does not match any predefined action, the assistant informs the user accordingly.</li>
    </ul>
  </li>
</ul>

<h2>✨ Key Features</h2>

<ul>
  <li><strong>Voice Customization:</strong> Users can select among different available voices via <code>pyttsx3</code>.</li>
  <li><strong>Command Flexibility:</strong> The program is designed to be easily extendable with new functionalities.</li>
  <li><strong>Security:</strong> Using <code>smtplib</code> requires sensitive information (e.g., email credentials). These must be handled securely to avoid risks.</li>
  <li><strong>Improved Recognition:</strong> Automatic background noise adjustment and Google Speech Recognition enhance accuracy.</li>
</ul>

<h2>🚧 Limitations & Possible Improvements</h2>

<ul>
  <li><strong>Multilingual Support:</strong> Currently optimized for French commands. Future development could add multilingual capabilities.</li>
  <li><strong>SMTP Error Handling:</strong> Email functionality could include more robust error checks (e.g., validating email addresses or handling connection issues).</li>
  <li><strong>Additional APIs:</strong> Third-party APIs (weather, news, smart home) could enrich functionality.</li>
  <li><strong>Graphical Interface:</strong> A GUI could be implemented to improve user experience.</li>
</ul>

<h2>✅ Conclusion</h2>

<p>
  This voice assistant provides a solid foundation for automated voice interaction systems. While it currently performs simple tasks, it offers considerable potential for further enhancements in personalization, security, and advanced capabilities.
</p>

<h2>🧰 Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>pyttsx3</li>
  <li>SpeechRecognition</li>
  <li>Wikipedia</li>
  <li>pywhatkit</li>
  <li>smtplib</li>
</ul>

<hr/>

<p align="center">
  <img src="https://media.giphy.com/media/hqU2KkjW5bE2v2Z7Q2/giphy.gif" width="300" alt="Futuristic Computer"/>
</p>

<h1 align="center">🛒 Retail Sales Forecasting</h1>

<p align="center">
  <em>Machine learning regression to predict product sales using tabular historical data and extensive feature engineering</em>
</p>

<hr/>

<h2>🎯 Objective</h2>

<p>
  This project aims to build a predictive model capable of estimating daily sales volumes for various products in retail stores across India. Leveraging detailed date-based and domain-specific feature engineering, the model provides accurate sales forecasts to support better inventory management and business planning.
</p>

<h2>📊 Dataset Overview</h2>

<ul>
  <li><strong>Original Data Format:</strong> Tabular dataset with 4 columns:
    <ul>
      <li><code>date</code>: Date of the sale record</li>
      <li><code>store</code>: Store identifier (point of sale)</li>
      <li><code>item</code>: Product identifier</li>
      <li><code>sales</code>: Sales volume</li>
    </ul>
  </li>
  <li><strong>Data Period:</strong> 2013 to 2017</li>
</ul>

<h2>⚙️ Processing Pipeline</h2>

<ul>
  <li><strong>Date Feature Engineering:</strong>
    <ul>
      <li>Extracted <code>year</code>, <code>month</code>, and <code>day</code> from <code>date</code></li>
      <li>Derived <code>day_name</code> (weekday names) based on the 2013–2017 calendar</li>
      <li>Created <code>weekday</code> feature:
        <ul>
          <li><code>0</code> for weekend days (Friday, Saturday, Sunday)</li>
          <li><code>1</code> for weekdays</li>
        </ul>
      </li>
      <li>Added <code>friday</code> flag indicating Indian holidays (1 = holiday, 0 = not holiday)</li>
      <li>Created <code>season</code> column to identify Winter, Summer, Spring, and Autumn</li>
    </ul>
  </li>
  <li><strong>Business Rationale for Temporal Features:</strong>
    <ul>
      <li>Week start and week end were identified to account for workload and consumption shifts (some products sell more at the start or end of the week)</li>
      <li>Month end periods were flagged, since financial constraints in families can lower consumption of specific products</li>
      <li>Holiday flags were created because consumption patterns increase for some products during national holidays</li>
      <li>Seasonality was added because certain products—especially fresh produce—are highly seasonal</li>
    </ul>
  </li>
  <li><strong>Historical Sales Features:</strong>
    <ul>
      <li>Generated lag features shifting sales by 30 days, 60 days, and more</li>
      <li>Applied higher weights to more recent sales (last 30 days) to reflect the stability and inertia of retail sales trends</li>
    </ul>
  </li>
  <li><strong>Missing Values Handling:</strong>
    <ul>
      <li>Filled <code>NaN</code> values in lag features with 0 to ensure robust training</li>
    </ul>
  </li>
  <li><strong>Encoding and Normalization:</strong>
    <ul>
      <li>Transformed categorical features such as <code>day_name</code> and <code>season</code> into numeric representations</li>
      <li>Normalized all numeric features to improve model convergence</li>
    </ul>
  </li>
  <li><strong>Data Split:</strong>
    <ul>
      <li>80% of the data for training</li>
      <li>20% for testing and evaluation</li>
    </ul>
  </li>
</ul>

<h2>🧠 Model Architecture</h2>

<ul>
  <li><strong>Regressor:</strong> Random Forest Regressor</li>
  <li><strong>Training Data:</strong> Engineered features excluding the <code>sales</code> column (target variable)</li>
</ul>

<h2>🛠️ Training Configuration</h2>

<ul>
  <li><strong>Model:</strong> sklearn RandomForestRegressor</li>
  <li><strong>Evaluation Metrics:</strong>
    <ul>
      <li>Mean Squared Error (MSE)</li>
      <li>R² Score</li>
    </ul>
  </li>
</ul>

<h2>✅ Results</h2>

<ul>
  <li><strong>Test Set MSE:</strong> 7.233264230948692e-26</li>
  <li><strong>R² Score:</strong> 1.0</li>
</ul>

<p>
  These results indicate perfect prediction accuracy on the test set, demonstrating the high predictive power of the model with engineered temporal and seasonal features.
</p>

<h2>🚀 Applications</h2>

<p>
  The model can be integrated into retail analytics platforms to:
  <ul>
    <li>Forecast demand per product and store</li>
    <li>Optimize inventory planning</li>
    <li>Support dynamic pricing strategies</li>
  </ul>
</p>

<h2>🧰 Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>Pandas</li>
  <li>scikit-learn</li>
  <li>Numpy</li>
</ul>

<hr/>

<p align="center">
  <img src="https://media.giphy.com/media/3oz8xKaR836UJOYeOc/giphy.gif" width="300" alt="Keep Going"/>
</p>
