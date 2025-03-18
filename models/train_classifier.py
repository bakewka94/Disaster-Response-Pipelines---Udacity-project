import sys
import pandas as pd
import pickle
import re
import nltk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

nltk.download("punkt")
nltk.download("wordnet")


def load_data(database_filepath):
    """
    Load data from SQLite database and split into X (features) and Y (labels).

    Args:
    database_filepath (str): Filepath to the SQLite database.

    Returns:
    X (pd.Series): Disaster messages.
    Y (pd.DataFrame): Corresponding categories.
    category_names (list): Names of category columns.
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("messages", engine)  # Ensure correct table name
    X = df["message"].fillna("")  # Handle missing values
    Y = df.iloc[:, 5:]  # Exclude non-label columns
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes and lemmatizes text data.

    Args:
    text (str): The message text to process.

    Returns:
    List[str]: A list of cleaned tokens.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  # Normalize text
    tokens = word_tokenize(text)  # Tokenize words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word).strip() for word in tokens]

    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline with GridSearchCV for tuning.

    Returns:
    model: GridSearchCV optimized classifier.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=tokenize)),  # Use TF-IDF directly
        ("clf", MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Define hyperparameters for tuning
    param_grid = {
        "clf__estimator__n_estimators": [10, 50],
        "clf__estimator__min_samples_split": [2, 4]
    }

    model = GridSearchCV(pipeline, param_grid, cv=2, verbose=2, n_jobs=-1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the trained model and prints classification reports.

    Args:
    model: Trained classifier.
    X_test (pd.Series): Test messages.
    Y_test (pd.DataFrame): True labels.
    category_names (list): Names of category columns.
    """
    Y_pred = model.predict(X_test)

    for i, category in enumerate(category_names):
        print(f"Category: {category}\n")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        print("-" * 60)


def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.

    Args:
    model: Trained classifier.
    model_filepath (str): Filepath to save the model.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model saved as {model_filepath}")


def main():
    """
    Execute the ML pipeline: load data, train model, evaluate, and save.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print(f"Loading data...\n    DATABASE: {database_filepath}")
        X, Y, category_names = load_data(database_filepath)

        # Debugging: Check sample data format
        print("Sample X_train (before splitting):", X.head())
        print("Data Type of X:", type(X))

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print(f"Saving model...\n    MODEL: {model_filepath}")
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print("Please provide the filepath of the disaster messages database "
              "as the first argument and the filepath of the pickle file to "
              "save the model to as the second argument. \n\nExample: python "
              "train_classifier.py ../data/DisasterResponse.db classifier.pkl")


if __name__ == "__main__":
    main()
