# Module to preprocess the title and the description
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from scripts.params import *


# Standardize the title and create
# ML & DL
def clean_title(title, keywords):
    # Split the title at the first hyphen
    parts = title.split(' - ')

    # Check if there are at least two parts and if the second part is a keyword
    if len(parts) > 1 and parts[1].strip() in keywords:
        return parts[0].strip()  # Return only the first part without trailing spaces
    else:
        return title.strip()  # Return the original title without trailing spaces

def analyze_and_clean_title(title):
    """
    Args:
        title (_type_): _description_

    Returns:
        _type_: _description_
    """
    keywords = ["Remote", "Part Time", "Full Time", "Earn up to $20.80/hr", "2nd Shift",
    "Contract", "Healthcare", "1st Shift", "Entry Level", "Telemetry",
    "RN", "Automotive", "Night Shift", "Retail", "Travel"]

    # Lowercasing for consistency
    title = title.lower()
    # Removing coded tags like "#BAS - "
    title = re.sub(r'^#.*?\s-\s', '', title)
    title = re.sub(r'(Part-Time)', '', title)
    # Remove string that without special character doesn't mean anything
    title.replace("C#", "").replace(".net", "").strip()
    # Split the title at the first hyphen and take the first part
    title = clean_title(title, keywords)
    # Removing non-alphabetic and non-space characters (like numbers and special characters)
    title = re.sub(r'[^a-zA-Z\s]', '', title)
    # Removing common prefixes/suffixes (like senior, junior, etc.)
    title = re.sub(r'\bsr\.*\b|\bsenior\b|\bjr\.*\b|\bjunior\b|\biii\b|\bii\b|\bi\b|\biv\b|\bstaff\b|\bdirector\b|\bassistant\b|\bassociate\b|\blead\b|\binternship\b', '', title)
    # Stripping extra spaces
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def standardize_job_title(title):
    # Define regex patterns
    data_scientist_pattern = re.compile(r'data\s+scientists?', re.IGNORECASE)
    data_analyst_pattern = re.compile(r'data\s+(and\s+)?anal(ytics?|ys[ti]s|ist)', re.IGNORECASE)
    data_engineer_pattern = re.compile(r'data\s+engineers?', re.IGNORECASE)
    solution_engineer_pattern = re.compile(r'solution\s+(engineer?|architect?)?', re.IGNORECASE)
    customer_success_pattern = re.compile(r'customer+(success|support|engagement|service|engineer?)?', re.IGNORECASE)

    # Check and replace with standardized titles
    if data_scientist_pattern.search(title):
        return 'data scientist'
    elif data_analyst_pattern.search(title):
        return 'data analyst'
    elif data_engineer_pattern.search(title):
        return 'data engineer'
    elif solution_engineer_pattern.search(title):
        return 'solutions engineer'
    elif customer_success_pattern.search(title):
        return 'customer success manager'
    else:
        return title  # Return original title if no match

def replace_title_with_target(title):
    # Convert the title to lower case for case-insensitive matching
    title_lower = title.lower()

    # Check if the title matches any of the target job titles
    for target_title in TARGET_JOB_TITLES:
        if target_title in title_lower:
            return target_title  # Replace with the target title
    return title  # Return the original title if no match found

# Function to apply grouping
def group_titles(title):
    return GROUPING_JOB_DICT.get(title, title)

# any(keyword in title for keyword in ['engineer', 'developer', 'programmer'])
def categorize_job(title):
    title = title.lower()
    if any(keyword in title for keyword in ['software', 'developer', 'programmer']):
        return 'Development'
    elif 'engineer' in title:
        return 'Engineer'
    elif 'data' in title:
        return 'Data'
    elif any(keyword in title for keyword in ['product', 'design']):
        return 'Product'
    elif any(keyword in title for keyword in ['consultant', 'advisor', 'solution', 'analyst']):
        return 'Consultant'
    elif any(keyword in title for keyword in ['sales', 'account', 'business', 'commercial']):
        return 'Sales'
    elif any(keyword in title for keyword in ['manager', 'director', 'chief', 'officer']):
        return 'Management'
    else:
        return 'Other'

# CLEAN DESCRIPTION
def remove_stopwords(sentence):
    stop_words = {'a','about','above','after','again','ain','all','am','an','and','any','are','aren','as','at','be','because','been','before','being','below','between','both','but','by','can','couldn','d','did','didn','do','does','doesn','doing','don','down','during','each','few','for','from','further','had','hadn','has','hasn','have','haven','having','he','her','here','hers','herself','him','himself','his','how','i','if','in','into','is','isn','it',"it's",'its','itself','just','ll','m','ma','me','mightn',"mightn't",'more','most','mustn',"mustn't",'my','myself','needn','nor','now','o','of','off','on','once','only','or','other','our','ours','ourselves','out','over','own','re','s','same','shan',"shan't",'she',"she's",'should',"should've",'shouldn','so','some','such','t','than','that',"that'll",'the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too','under','until','up','ve','very','was','wasn','we','were','weren','what','when','where','which','while','who','whom','why','will','with','won','wouldn','y','you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves'}

    tokens = nltk.word_tokenize(sentence)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words ]
    return ' '.join(filtered_tokens)

def clean_description(sentence):
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
    #sentence = re.sub(r"[`.]{2,}|-{2,}", "", sentence)
    return sentence

def remove_title(title, description):
    title = title.lower()
    new_desc = description.replace(title,'').strip()
    return new_desc

# Generate ML df
def package_category_df(df: pd.DataFrame, data_dir_clean):
    df['category'] = df['title'].apply(categorize_job)

    print('CATEGORY')
    df_clean_category = df[df['category'] != 'Other']
    print(df_clean_category['category'].value_counts())
    print('NB OF LINES KEPT:')
    print(df_clean_category['category'].value_counts().sum())

    df_with_category_ml = df[df['category'] != 'Other'].copy()
    df_with_category_ml.drop(columns=['title', 'cleaned_title', 'source'], inplace=True)
    df_with_category_ml.to_csv(f'{data_dir_clean}/df_category_all.csv', index=False)
    print('CATEGORY / DESCRIPTION CREATED')

    return df_with_category_ml


# Generate DL df
def package_job_title_df(df: pd.DataFrame, data_dir_clean):
    df['jobs_to_keep'] = df['cleaned_title'].apply(lambda x: x in FINAL_JOB_TITLES)

    print('RATIO BETWEEN JOB TITLE THAT ARE CLASSIFIED (TRUE) AND THE OTHER (FALSE)')
    print(df['jobs_to_keep'].value_counts())

    # Creating a new column to indicate if the job title is one of the target job titles to keep
    df_with_job_title_dl = df[df['jobs_to_keep']].copy()
    print('JOB TITLE')
    print(df_with_job_title_dl['cleaned_title'].value_counts())
    print('NB OF LINES KEPT:')
    print(df_with_job_title_dl['cleaned_title'].value_counts().sum())

    df_with_job_title_dl['title'] = df_with_job_title_dl['cleaned_title']
    df_with_job_title_dl.drop(columns=['cleaned_title', 'jobs_to_keep', 'source'], inplace=True)
    df_with_job_title_dl.to_csv(f'{data_dir_clean}/df_job_title_all.csv', index=False)
    print('CLEAN TITLE / DESCRIPTION CREATED')

    return df_with_job_title_dl

if __name__ == "__main__":
    import os
    # Get the full dataframe
    cur_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(cur_dir)
    data_dir_raw = os.path.join(project_dir, "../data/raw_data")
    data_dir_clean = os.path.join(project_dir, "../data/clean_data")

    print('READ RAW PARQUET')
    df = pd.read_csv(f'{data_dir_raw}/final_dataset_50k.csv')
    df.dropna(inplace=True)

    # Toutes les fonctions descriptions
    print('CLEAN DESCRIPTION: START')
    df['description'] = df['description'].apply(remove_stopwords)
    df['description'] = df['description'].apply(clean_description)
    df['description'] = df.apply(lambda row: remove_title(row['title'] , row['description']), axis=1)
    print('CLEAN DESCRIPTION: DONE')

    # Apply the categorization function to the dataframe
    print('CLEAN TITLE: START')
    df['cleaned_title'] = df['title'].apply(analyze_and_clean_title)
    df['cleaned_title'] = df['cleaned_title'].apply(standardize_job_title)
    df['cleaned_title'] = df['cleaned_title'].apply(replace_title_with_target)
    df['cleaned_title'] = df['cleaned_title'].apply(group_titles)
    print('CLEAN TITLE: DONE')

    print('CREATE NEW PARQUET FOR ML AND DL POST-PROCESSING')
    package_category_df(df, data_dir_clean)
    package_job_title_df(df, data_dir_clean)

    print('ALL DONE')
