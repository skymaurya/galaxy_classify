# galaxy_classify

# SDSS Galaxy Classification using Machine Learning Techniques

This project implements a binary classification model to classify galaxies from the Sloan Digital Sky Survey (SDSS) dataset into two categories: starforming and starburst. It utilizes three machine learning algorithms: Decision Tree, Logistic Regression, and Random Forest. The model is deployed as a web application using Flask, allowing users to input galaxy features and receive predictions.

## Project Description

The SDSS dataset contains a wealth of information about galaxies, including their spectral properties, morphology, and luminosity. This project aims to leverage this data to build a machine learning model capable of accurately classifying galaxies based on their characteristics. The model is trained on a subset of the SDSS dataset and evaluated using various metrics to assess its performance.

## Technology Stack

* **Python:** The primary programming language used for model development and web application creation.
* **python libraries:**
     pandas, numpy, seaborn,matplotlib.pyplot , scikit-learn,imblearn.
* **Flask:** A lightweight web framework for building the web application.
* **HTML:** Used to create the user interface for inputting galaxy features.

## Key Features

* **Binary Classification:** The model classifies galaxies into two categories: starforming and starburst.
* **User Input:** The web application allows users to input 10 galaxy features.
* **Prediction:** The model predicts the galaxy type based on the user's input.
* **Web Interface:** A simple and user-friendly web interface is provided for interaction.

## Project Overview

The Galaxy Classification Project aims to predict galaxy subclasses using machine learning techniques. The project involves data collection, preparation, exploratory data analysis (EDA), model building, performance testing, and deployment. The final model is integrated with a web framework to provide real-time predictions based on user input.

## Project Flow

1. **User Interaction**: Users interact with a web-based UI to input data.
2. **Model Analysis**: The input data is analyzed using a machine learning model integrated into the web application.
3. **Prediction Display**: The model’s prediction is displayed to the user on the UI.

## Detailed Steps

### 1. Data Collection & Preparation

#### Collect the Dataset
The dataset for this project is sourced from the Sloan Digital Sky Survey (SDSS). It consists of photometric image data for galaxies, with 100,000 rows and two primary galaxy subclasses: 'STARFORMING' and 'STARBURST'. The dataset can be downloaded from Kaggle.

#### Data Preparation
- **Handling Missing Values**: Missing values are identified and appropriately handled.
- **Data Type Conversion**: The 'subclass' column is converted from object to integer using ordinal encoding.
- **Feature Selection**: The dataset is refined by selecting relevant features for model training.

### 2. Exploratory Data Analysis (EDA)

#### Descriptive Statistics
- **Purpose**: To understand the fundamental characteristics of the data.
- **Tools**: Pandas `describe()` function to summarize statistics like mean, standard deviation, and percentiles.

#### Visual Analysis
- **Purpose**: To visually explore data and identify patterns, trends, and outliers.
- **Tools**: Seaborn and Matplotlib for creating visualizations like box plots and heatmaps.

### 3. Model Building

#### Training the Model
Different machine learning algorithms are trained to find the best model:
- **Decision Tree Classifier**: A simple and interpretable model.
- **Logistic Regression**: A statistical method for binary classification.
- **Random Forest Classifier**: An ensemble method that improves accuracy through multiple decision trees.

#### Testing the Model
- **Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix are used to evaluate model performance.
- **Comparison**: Models are compared based on these metrics to select the best-performing model.

### 4. Model Evaluation

After training and testing various models, the performance metrics for each model were compared to determine the best performing algorithm. The table below summarizes the precision, recall, f1-score, and accuracy of each model:

|         **Model**        | **Precision Class 0** | **Precision Class 1** | **Recall Class 0** | **Recall Class 1** | **f1-score Class 0** | **f1-score Class 1** | **Accuracy** | **Overall Performance** |
|:------------------------:|:---------------------:|:---------------------:|:------------------:|:------------------:|:--------------------:|:--------------------:|:------------:|:-----------------------:|
| Decision Tree Classifier |          0.80         |          0.80         |        0.79        |        0.80        |         0.80         |         0.80         |  **0.79931** |      **Average**        |
| Logistic Regression      |          0.80         |          0.82         |        0.82        |        0.79        |         0.81         |         0.81         |  **0.80848** |       **Good**          |
| Random Forest Classifier |          0.83         |          0.81         |        0.81        |        0.83        |         0.82         |         0.82         |  **0.82068** |       **Best**          |

#### Analysis

- **Decision Tree Classifier**: 
  - **Strengths**: Good balance in precision for both classes, making it a reliable choice.
  - **Weaknesses**: Lower recall for Class 0, indicating that the model misses some instances of Class 0.
  - **Overall Performance**: The model has the lowest overall accuracy compared to others.

- **Logistic Regression**:
  - **Strengths**: Good balance in precision for both classes, making it a reliable choice.
  - **Weaknesses**: Lower recall for Class 1, suggesting it is less effective at identifying Class 1 instances.
  - **Overall Performance**: Moderate overall accuracy, making it a decent choice but not the best.

- **Random Forest Classifier**:
  - **Strengths**: Highest precision for class 0 and highest recall for Class 1.
  - **Weaknesses**: Slightly lower precision for Class 1 compared to Logistic Regression.
  - **Overall Performance**: Best overall accuracy, indicating it performs best across all metrics.

### 5. Model Deployment

#### Save the Best Model
- **File**: `RF1.pkl` - The best-performing Random Forest model is saved using Python’s `pickle` module. This avoids the need to retrain the model and allows for future use.

#### Integrate with Web Framework
- **Web Application**: Built using Flask to create a user-friendly interface.
- **HTML Pages**: 
  - `index.html` - The main page where users input their data.
  - `inner-page.html` - The page displaying the prediction results.

### Web Framework Integration

#### Building HTML Pages
- **Files**: 
  - `index.html` - Contains the form for user input.
  - `inner-page.html` - Displays the prediction results.

#### Building Server-Side Script
- **File**: `app.py` - Contains Flask routes for rendering HTML pages and handling user inputs.
- **Functionality**: 
  - Load the saved model (`RF.pkl`).
  - Render HTML pages.
  - Retrieve input values from the UI and make predictions using the model.

#### Running the Web Application
1. **Setup**: Open VS Code and navigate to the folder containing `app.py`.
2. **Command**: Run `python app.py` to start the Flask server.
3. **Access**: Open a web browser and go to `http://127.0.0.1:5000` to interact with the web application.
4. **Usage**: Enter input values on the `index.html` page and view predictions on the `inner-page.html` page.

## Installation Instructions

To set up the environment and run the Galaxy classificastion project  on your system follow these steps:

1. Clone the Repository
First, clone the repository containing the notebook to your local machine:

     git clone <repository-url>
     cd <repository-directory>


2. Install VS Code  if not already installed.
Download and install Visual Studio Code.

3. Install Python Extension.
Open VS Code and install the Python extension from the marketplace. To do this:
    1. Open VS Code.
    2.Go to the Extensions view by clicking on the Extensions icon in the Activity Bar on the side of the window.
    3. Search for "Python" and install the extension provided by Microsoft.

  
4. Create a Virtual Environment.
It is recommended to use a virtual environment to manage dependencies. 
Create and activate a virtual environment using venv:
    a.python -m venv venv
      source venv/bin/activate  
    b.On Windows use `venv\Scripts\activate`

5. Install Dependencies.
Install the required libraries using pip:

      pip install -r requirements.txt


6. Open the Project in VS Code.
Open the repository folder in VS Code:

    Go to File > Open Folder....
    Select the cloned repository folder.


7. Select the Python Interpreter.
Make sure to select the Python interpreter for your virtual environment:

    1. Press Ctrl+Shift+P to open the command palette.
    2. Type Python: Select Interpreter and select the interpreter from your virtual environment 
      (it should be something like venv/bin/python or venv\Scripts\python.exe).

8. Install Jupyter Extension.
Install the Jupyter extension from the marketplace. To do this:
    Go to the Extensions view by clicking on the Extensions icon in the Activity Bar.
    Search for "Jupyter" and install the extension provided by Microsoft.

9. Run the Jupyter Notebook.
    Open galxy.ipynb in VS Code.
    Click on the Run button at the top of the notebook or press Shift+Enter to run individual cells.

10. To Run the web application.
    1.open and run python 'app.py' file to start Flask Sever.In terminal click on IP address`http://127.0.0.1:5000` to       interact with web app.
    2.in web app click on the Get Started button to give input values to model for classification.
    3.After filling values click on the 'Submit' Button, to see what model classify on the basis of inputs given.

