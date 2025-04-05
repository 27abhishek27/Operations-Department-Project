# Operations Department Project

## 📌 Overview

This project focuses on developing a deep learning model to detect **COVID-19** and differentiate it from other types of pneumonia (viral and bacterial) using chest X-ray images. The model aims to assist healthcare professionals in accurately diagnosing patients based on radiographic data.

## 📂 Project Structure

```
├── Dataset/                             # Directory containing the dataset of X-ray images
├── Operations Department Project png/   # Visualizations and plots generated during analysis
├── Operations_Department_png/           # Additional visual assets
├── Test/                                # Directory for testing scripts and related files
├── Operations_Department_Project.ipynb  # Jupyter Notebook with data analysis and model implementation
├── README.md                            # Project documentation
└── .gitattributes                       # Git configuration attributes
```

## 📊 Dataset

The dataset for this project is compiled from the following sources:

- **COVID-19 X-ray Images**: Sourced from [COVID-19 Radiography Database](https://github.com/ieee8023/covid-chestxray-dataset).
- **Pneumonia X-ray Images**: Obtained from [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

Each class in the dataset contains 133 X-ray images, categorized as follows:

- **0 - COVID-19**: X-ray images of patients diagnosed with COVID-19.
- **1 - Normal**: X-ray images of healthy individuals.
- **2 - Viral Pneumonia**: X-ray images of patients with viral pneumonia.
- **3 - Bacterial Pneumonia**: X-ray images of patients with bacterial pneumonia.

## 🚀 Installation

### 1️⃣ Clone the repository:

```bash
git clone https://github.com/27abhishek27/Operations-Department-Project.git
cd Operations-Department-Project
```

### 2️⃣ Install dependencies:

Ensure you have the following Python packages installed:

- `tensorflow`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `opencv-python`

You can install them using pip:

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn opencv-python
```

## 🔍 Methodology

### 1. **Data Preprocessing**

- **Data Collection**: Aggregated X-ray images from multiple sources to create a balanced dataset.
- **Image Augmentation**: Applied techniques such as rotation, scaling, and flipping to increase dataset diversity and improve model generalization.
- **Normalization**: Standardized pixel values to a common scale to enhance model performance.

### 2. **Exploratory Data Analysis (EDA)**

- **Class Distribution**: Analyzed the number of images per category to ensure balanced representation.
- **Sample Visualization**: Displayed sample images from each class to understand visual differences and similarities.

### 3. **Model Building**

- **Convolutional Neural Network (CNN) Architecture**: Designed a CNN model with multiple convolutional and pooling layers to extract features from X-ray images.
- **Activation Functions**: Utilized ReLU activation for hidden layers and softmax for the output layer to handle multi-class classification.
- **Compilation**: Compiled the model with categorical cross-entropy loss and an optimizer like Adam.

### 4. **Model Training and Evaluation**

- **Data Splitting**: Divided the dataset into training and validation sets to monitor model performance.
- **Training**: Trained the CNN model on the training data with techniques like early stopping to prevent overfitting.
- **Evaluation Metrics**: Assessed model performance using accuracy, precision, recall, F1-score, and confusion matrices.

## 📊 Visualizations:

Here are some visualizations from the project:

![alt text](https://github.com/27abhishek27/Operations-Department-Project/blob/main/Operations%20Department%20Project%20png/result.png)
![alt text](https://github.com/27abhishek27/Operations-Department-Project/blob/main/Operations%20Department%20Project%20png/subplot.png)
![alt text](https://github.com/27abhishek27/Operations-Department-Project/blob/main/Operations%20Department%20Project%20png/tarining%20accuracy%20and%20loss.png)
![alt text](https://github.com/27abhishek27/Operations-Department-Project/blob/main/Operations%20Department%20Project%20png/validation%20accuracy.png)
![alt text](https://github.com/27abhishek27/Operations-Department-Project/blob/main/Operations%20Department%20Project%20png/validation%20loss.png)

## 🛠️ Technologies Used

- **Python**
- **TensorFlow & Keras**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **OpenCV**
- **Jupyter Notebook**

## 📌 Future Improvements

- **Hyperparameter Tuning**: Implement techniques like GridSearchCV to optimize model parameters for better accuracy.
- **Dataset Expansion**: Incorporate more X-ray images to enhance model robustness and generalization.
- **Model Deployment**: Develop a web-based interface to deploy the model for real-time diagnosis assistance.
- **Explainability**: Utilize tools like Grad-CAM to visualize which regions of the X-ray images influence model decisions.
