# Disaster Response Pipeline Project

This project is part of the Udacity Data Science Nanodegree and involves building a machine learning pipeline to classify disaster response messages. The web application allows emergency workers to input messages and receive categorized predictions to help in disaster relief efforts.

## Table of Contents
- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Instructions](#instructions)
- [ML Pipeline](#ml-pipeline)
- [Web App](#web-app)
- [Screenshots](#screenshots)
- [Acknowledgments](#acknowledgments)

## Project Overview
The project follows an end-to-end machine learning pipeline:
1. **Data Processing**: ETL pipeline extracts, transforms, and loads messages from disaster response sources.
2. **Machine Learning Pipeline**: A classifier is trained to categorize messages into relevant emergency response categories.
3. **Web Application**: A Flask-based interface allows users to input messages and receive category predictions.

## File Structure
```
Disaster-Response-Pipeline
│─── app
│    ├── templates
│    │    ├── go.html
│    │    ├── master.html
│    ├── run.py  # Runs the web app
│
│─── data
│    ├── disaster_messages.csv  # Raw messages dataset
│    ├── disaster_categories.csv  # Categories dataset
│    ├── process_data.py  # ETL pipeline script
│    ├── DisasterResponse.db  # Processed SQLite database
│
│─── models
│    ├── train_classifier.py  # Machine learning pipeline script
│    ├── classifier.pkl  # Trained ML model
│
│─── README.md  # Project documentation
│─── requirements.txt  # Dependencies
```

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/bakewka94/Disaster-Response-Pipelines---Udacity-project.git
   cd Disaster-Response-Pipelines---Udacity-project
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download or use the provided dataset in `data/`, then process it:
   ```bash
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```
4. Train the model:
   ```bash
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```
5. Run the web app:
   ```bash
   python app/run.py
   ```
   Open the app in your browser at `http://127.0.0.1:3000/`

## ML Pipeline
The machine learning pipeline consists of:
1. **Text processing** (tokenization, lemmatization, and TF-IDF transformation).
2. **Feature extraction** (converting messages into numerical features).
3. **Model training** (using `scikit-learn` pipeline with GridSearchCV for tuning).
4. **Evaluation** (precision, recall, F1-score metrics).

## Web App
The Flask web app allows users to:
- Enter a message and see its predicted categories.
- View visualizations of message data distribution.

## Screenshots
### Web Application Interface:
![Web App Screenshot](https://github.com/bakewka94/Disaster-Response-Pipelines---Udacity-project/raw/main/The%20web%20app%20screenshot.png)

### Classification Example:
![Classification Example](https://github.com/bakewka94/Disaster-Response-Pipelines---Udacity-project/raw/main/Classification%20example%20screenshot.png)


## Acknowledgments
This project is part of the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
Dataset provided by [Figure Eight](https://appen.com).
