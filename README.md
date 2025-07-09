
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
  <img src="https://media.giphy.com/media/3oKIPsx2edrJp2g2s8/giphy.gif" width="300" alt="Data Visualization"/>
</p>
