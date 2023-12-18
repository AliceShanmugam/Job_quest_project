import os
import pandas as pd

from scripts.preprocessor.preprocessing_dataset import remove_stopwords, clean_description, remove_title
from scripts.preprocessor.preprocessing_dataset import analyze_and_clean_title, standardize_job_title, replace_title_with_target, group_titles
from scripts.preprocessor.preprocessing_dataset import package_category_df, package_job_title_df

from scripts.preprocessor.preprocessing_deep import train_val_split

from scripts.model.first_deep import init_model, fit_model
from scripts.model.registry import save_model

# -------------------
# Preprocesser le dataset --> Dataframe clean
# -------------------
def preprocess(type: str, raw_df) -> pd.DataFrame:
    # -------------------
    # LOAD DATASET
    # -------------------
    cur_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(cur_dir)
    data_dir_raw = os.path.join(project_dir, "data/raw_data")
    data_dir_clean = os.path.join(project_dir, "data/clean_data")

    print('READ RAW PARQUET')
    df = pd.read_csv(f'{data_dir_raw}/final_dataset_50k.csv')
    df.dropna(inplace=True)

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

    print('CREATE NEW PARQUET FOR DL POST-PROCESSING')
    clean_df_DL = package_job_title_df(df, data_dir_clean)
    return clean_df_DL

# -------------------
# DEEP LEARNING - Prepare le dataframe pour entrainer le model en deep --> X_train, y_train, X_val, y_val
# -------------------
def train_deep(
        clean_df_DL,
        batch_size = 32,
        patience = 3
    ) -> float:
    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights
    """
    print("⭐️ Use case: train")

    X_train, X_val, y_train, y_val = train_val_split(clean_df_DL) # Stratify

    # Train model using `model.py`
    print('Init Model')
    model = init_model(X_train, y_train, "dense")

    print('Start fitting Model')
    model, history = fit_model(
        model,
        X_train,
        y_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val, y_val)
    )

    print(history)

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")




if __name__ == '__main__':
    clean_df_DL = preprocess(type="dl")
    train_deep(clean_df_DL)
