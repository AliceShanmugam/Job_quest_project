import os
import pickle
import time
import pandas as pd

from scripts.preprocessor.preprocessing_dataset import remove_stopwords, clean_description, remove_title
from scripts.preprocessor.preprocessing_dataset import analyze_and_clean_title, standardize_job_title, replace_title_with_target, group_titles
from scripts.preprocessor.preprocessing_dataset import package_category_df

from scripts.preprocessor.preprocessing_ml import train_test_split_df, vectorize_X, target_encode

from scripts.model.MNB_model import init_model, fit_model
from scripts.model.registry import save_model

from scripts.params import *

# -------------------
# Preprocesser le dataset --> Dataframe clean
# -------------------

def preprocess(type: str) -> pd.DataFrame:
    # -------------------
    # LOAD DATASET
    # -------------------
    cur_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(cur_dir)
    data_dir_raw = os.path.join(project_dir, "data/raw_data")
    data_dir_clean = os.path.join(project_dir, "data/clean_data")

    print('READ RAW PARQUET')
    df = pd.read_parquet(f'{data_dir_raw}/cleaning_data_eng_eng.parquet', engine='fastparquet')
    #df = pd.read_csv(f'{data_dir_clean}/top_10_jobs.csv')
    df.dropna(inplace=True)
    df.rename(columns={'job_description': 'description', 'job_title':'title'}, inplace=True)

    print('CLEAN DESCRIPTION: START')
    df['description'] = df['description'].apply(remove_stopwords)
    df['description'] = df['description'].apply(clean_description)
    df['description'] = df.apply(lambda row: remove_title(row['title'] , row['description']), axis=1)

    # Apply the categorization function to the dataframe
    print('CLEAN TITLE: START')
    df['cleaned_title'] = df['title'].apply(analyze_and_clean_title)
    df['cleaned_title'] = df['cleaned_title'].apply(standardize_job_title)
    df['cleaned_title'] = df['cleaned_title'].apply(replace_title_with_target)
    df['cleaned_title'] = df['cleaned_title'].apply(group_titles)

    print('CREATE NEW PARQUET FOR ML POST-PROCESSING')
    clean_df_ML = package_category_df(df, data_dir_clean)
    return clean_df_ML


# -------------------
# MACHINE LEARNING - Prepare le dataframe pour entrainer le model en machine learning --> X_train, y_train, X_val, y_val
# -------------------

def train_ml(clean_df_ML):

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights
    """
    print("⭐️ Use case: train")

    # Split test and train sets
    X_train, X_test, y_train, y_test = train_test_split_df(clean_df_ML)


    # Outputs of vectorize_X on X_train and X_test
    X_train_transformed, X_test_transformed, vectorizer = vectorize_X(X_train, X_test)

    # Save fitted ml vectorizer locally
    timestamp_vec = time.strftime("%Y%m%d-%H%M%S")
    ml_vectorizer_path = os.path.join(LOCAL_REGISTRY_PATH, "vectorizers_ml", f"ml_vectorizer_{timestamp_vec}.pickle")

    # save fitted ml vectorizer
    pickle.dump(vectorizer, open(ml_vectorizer_path, "wb"))


    # target_encode on y_train and y_test, label encode categories
    y_train_transformed, y_test_transformed, label_encoder = target_encode(y_train, y_test)

    # Save fitted ml vectorizer locally
    timestamp_vec = time.strftime("%Y%m%d-%H%M%S")
    ml_labelencoder_path = os.path.join(LOCAL_REGISTRY_PATH, "labelencoders_ml", f"ml_labelencoder_{timestamp_vec}.pickle")

    # save fitted ml vectorizer
    pickle.dump(label_encoder, open(ml_labelencoder_path, "wb"))


    # Train model using `model.py`
    print('Init Model')
    model = init_model()

    print('Start fitting Model')
    model_ml = fit_model(model, X_train_transformed, y_train_transformed)


    # Save model weight on the hard drive (and optionally on GCS too!)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models_ml", f"model_{timestamp}.pickle")
    # save model
    pickle.dump(model_ml, open(model_path, "wb"))

    print("✅ train() done \n")

if __name__ == '__main__':
    clean_df_ML = preprocess(type="ml")
    train_ml(clean_df_ML)
