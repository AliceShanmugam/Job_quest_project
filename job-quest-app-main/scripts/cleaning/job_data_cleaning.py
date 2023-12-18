# Basic Imports
import os
import re
import string
import html
import unicodedata
import pandas as pd

# Project import
from langdetect import detect
from scripts.cleaning.test_data_cleaning import running_tests

### Variables
# Path & CSV name
# Path
cur_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(cur_dir)
data_dir = os.path.join(project_dir, "data")

# CSV Name
glassdoor = "glassdoor.csv"
indeed = "indeed.csv"
linkedin = "linkedin.csv"
career_build = "career_builder.json"

# CSV Path
glassdoor_dataset = os.path.join(data_dir, glassdoor)
indeed_dataset = os.path.join(data_dir, indeed)
linkedin_dataset = os.path.join(data_dir, linkedin)
career_build_dataset = os.path.join(data_dir, career_build)

# Concat Path
datasets_path = [
    glassdoor_dataset,
    indeed_dataset,
    linkedin_dataset,
    career_build_dataset,
]


### Functions

# Get Data set
def get_data(path):
    # split file & ext
    file, ext = os.path.splitext(path)
    source = file.split("/")[-1]

    print("Please wait ... ")
    print(f"Get {source} dataset... ")

    if ext == ".json":
        df = pd.read_json(path)
        df["source"] = source
        return df

    df = pd.read_csv(path)
    df["source"] = source
    return df


# Set Header
def set_header_dataset(df):
    source = df.source[0]

    print("Please wait ... ")
    print(f"Get {source} dataset... ")

    if source == "indeed":
        return df[["Job Title", "Job Description", "source"]].rename(
            columns={"Job Title": "job_title", "Job Description": "job_description"}
        )
    if source == "glassdoor":
        return df[["header.jobTitle", "job.description", "source"]].rename(
            columns={
                "header.jobTitle": "job_title",
                "job.description": "job_description",
            }
        )
    if source == "linkedin" or source == "career_builder":
        return df[["title", "description", "source"]].rename(
            columns={"title": "job_title", "description": "job_description"}
        )


def remove_emoji_unicode(text):
    # Extended emoji pattern
    extended_emoji_pattern = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return extended_emoji_pattern.sub("", text)

def clean(df):
    # Drop NaN
    print("Dropping the NaN ...")
    df.dropna(inplace=True)
    print("✅")

    # lower case all strings
    print("Lower case ...")
    df["job_description"] = df["job_description"].apply(lambda x: x.lower())
    print("✅")

    # Normalize Unicode characters
    df["job_description"] = df["job_description"].apply(
        lambda x: unicodedata.normalize("NFKC", x)
    )

    # Unscape texts in html
    print("Unscaping HTML ...")
    df["job_description"] = df["job_description"].apply(lambda x: html.unescape(x))
    print("✅")

    # Clean tags
    print("Cleaning tags ...")
    df["job_description"] = df["job_description"].apply(
        lambda x: re.sub("<[^<>]*>", "", x)
    )
    print("✅")

    # Clean \n
    print("Cleaning return ...")
    df["job_description"] = df["job_description"].apply(lambda x: re.sub("\n", " ", x))
    print("✅")

    # Remove emoji
    print("Cleaning emojis ...")
    df["job_description"] = df["job_description"].apply(remove_emoji_unicode)
    print("✅")

    # Remove Extra Space
    print("Removing Extra Space ...")
    # Replace specific non-standard whitespace characters
    df["job_description"] = df["job_description"].apply(
        lambda x: x.replace("\xa0", " ")
    )

    # Use regex to remove extra spaces
    df["job_description"] = df["job_description"].apply(lambda x: re.sub("\s+", " ", x))
    df["job_description"] = df["job_description"].apply(lambda x: " ".split())
    print("✅")

    # Remove Digit
    print("Removing digits ...")
    df["job_description"] = df["job_description"].apply(
        lambda x: "".join([i for i in x if not i.isdigit()])
    )
    print("✅")

    # Remove punctuation
    PUNCT_TO_REMOVE = string.punctuation
    print("Removing Punctuation ...")
    df["job_description"] = df["job_description"].apply(
        lambda x: x.translate(str.maketrans("", "", PUNCT_TO_REMOVE))
    )
    print("✅")

    # Drop NaN
    print("Final Attention to the NaN...")
    df.dropna(inplace=True)
    print("✅")

    return df


def saving_df(df, filename="cleaning_data", file_type="parquet"):
    ext = ".parquet" if file_type == "parquet" else ".csv"

    full_filename = filename + ext

    # Check if the file exists and increment filename if it does
    i = 1
    while os.path.exists(full_filename):
        full_filename = f"{filename}_{i}{ext}"
        i += 1

    # Save the file based on the file type
    if file_type == "parquet":
        df.to_parquet(full_filename)
    else:
        df.to_csv(full_filename, index=False, encoding="utf-8")

    print(f"File saved as {full_filename}")


# Main
def main(df_path: list) -> pd.DataFrame:
    print("Get the data ... ")
    all_df = [get_data(df) for df in df_path]
    print("All Data fetch ✅ \n")

    print("Set the headers ... ")
    datasets = [set_header_dataset(df) for df in all_df]
    print("All headers set correctly ✅ \n")

    print("Combined the datasets ... ")
    combined_df = pd.concat(datasets, ignore_index=True)
    print("All dataset combined correctly ✅ \n")

    print("Cleaning the data ... ")
    clean_df = clean(combined_df)

    print("All Data is corretly clean ✅ \n")

    print("First Saving ... ")
    filename = saving_df(clean_df)
    print("Data save corretly ✅ \n")

    print("Extra Cleaning ... !!")
    df_cleaned = pd.read_parquet(filename)
    df_cleaned = running_tests(df_cleaned)
    print("Save the Clean_df ")

    saving_df(df_cleaned, filename="perfect_data")
    print("Data save corretly ✅ \n")

    return clean_df


if __name__ == "__main__":
    main(datasets_path)
