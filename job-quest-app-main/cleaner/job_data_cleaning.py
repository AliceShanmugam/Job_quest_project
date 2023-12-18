# Basic Imports
import os
import re
import string
import html
import unicodedata
import pandas as pd

# Project import
from langdetect import detect
from langdetect import lang_detect_exception
from test_data_cleaning import running_tests
from hammertime import Timer

### Variables
# Path & CSV name
# Path
cur_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(cur_dir))
data_dir = os.path.join(project_dir, "data", "job_dataset")

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

# Set the timer
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

def remove_accents(text):
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def remove_currency_symbols(text):
    currency_symbols_pattern = r'[\$\€\£\¥\₹\₩\₽]'
    return re.sub(currency_symbols_pattern, '', text)

def remove_extra_spaces(text):
    text = text.strip()
    extra_space_pattern = re.compile(r"  +")
    return extra_space_pattern.sub(' ', text)

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

    # Remove Currency
    print("Cleaning currency ...")
    df["job_description"] = df["job_description"].apply(remove_currency_symbols)
    print("✅")

    # Clean accents
    print("Cleaning accents ...")
    df["job_description"] = df["job_description"].apply(remove_accents)
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

    # Remove Extra Space
    print("Removing Extra Space ...")
    df["job_description"] = df["job_description"].apply(remove_extra_spaces)
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

    return full_filename

def is_english(df: pd.DataFrame):

    row_to_drop = []

    print('Identification Language ...')
    for idx, row in df.iterrows():
        try:
            language = detect(row['job_description'])
            if language != 'en':
                row_to_drop.append(idx)
        except lang_detect_exception.LangDetectException as e:
            print(f"Error, drop {e}")
            row_to_drop.append(idx)

        if idx % 1000 == 0:
            print(f'Text to drops : {len(row_to_drop)}')
            print(f"Step : {idx} / {df.shape[0]}")

    print(f'Dropping {len(row_to_drop)} rows ... ')
    df.drop(row_to_drop, inplace= True)

    return df


# Main
def data_cleaner(df_path: list) -> pd.DataFrame:
    print('Starting Process ... \n')
    print("Get the data ... ")
    all_df = [get_data(df) for df in df_path]
    print("All Data fetch ✅ \n")

    print("Setthe headers ... ")
    datasets = [set_header_dataset(df) for df in all_df]
    print("All headers set correctly ✅ \n")

    print("Combined the datasets ... ")
    combined_df = pd.concat(datasets, ignore_index=True)
    print("All dataset combined correctly ✅ \n")

    print("Cleaning the data ... ")
    clean_df = clean(combined_df)
    print("All Data is corretly clean ✅ \n")

    print("Saving clean data ... ")
    filename = saving_df(clean_df)
    print("Data save corretly ✅ \n")

    return clean_df, filename

def test_cleaning(filename):
    print("Extra Cleaning ... !!")
    clean_df = pd.read_parquet(filename)
    clean_df = running_tests(clean_df)

    print("Save the Clean_df ")
    filename = saving_df(clean_df, filename="perfect_data")
    print("Data save corretly ✅ \n")

    return clean_df, filename

def clean_language_function(filename):

    cleanest_df = pd.read_parquet(filename)
    eng_df = is_english(cleanest_df)

    print("Saving ... ")
    filename = saving_df(eng_df, filename=f"{filename.removesuffix('.parquet')}_eng")
    print("Data save corretly ✅ \n")
    return eng_df, filename

def list_parquet_files(directory):

    parquet_files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    return parquet_files

def choose_parquet_file():

    directory = input("Enter the directory path where .parquet files are stored: ")
    files = list_parquet_files(directory)

    if not files:
        print("No .parquet files found in the directory.")
        return None

    print("\nAvailable .parquet files:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")

    while True:
        choice = input("Enter the number of the file you want to choose (or 'exit' to cancel): ")
        if choice.lower() == 'exit':
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return os.path.join(directory, files[int(choice) - 1])
        else:
            print("Invalid choice. Please enter a valid number or 'exit'.")


def menu():
    while True:

        print("\nData Processing Menu")
        print("1. Clean Data")
        print("2. Test Data")
        print("3. Clean Language")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == '1':
            print("Cleaning Data...")

            time = Timer()
            time.start()
            data_cleaner(datasets_path)
            time.stop()

        elif choice == '2':
            print("Testing Data...")

            time = Timer()
            time.start()
            filename = choose_parquet_file()
            test_cleaning(filename)
            time.stop()

        elif choice == '3':
            print("Cleaning Language...")

            time = Timer()
            time.start()
            filename = choose_parquet_file()
            clean_language_function(filename)
            time.stop()

        elif choice == '4':
            print("Exiting...")

            break

        else:

            print("Invalid choice. Please enter a number between 1 and 4.")


if __name__ == "__main__":
    menu()
