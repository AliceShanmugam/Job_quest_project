# Main.py
import numpy as np
import pandas as pd
import re
import html
import string

# use regex to remove html and css in job description
def clean_code(text):
    cleaned_html_text = re.sub('<[^<>]*>','', text)
    cleaned_jd = re.sub('job description\n','', cleaned_html_text)
    cleaned_text = re.sub('\n',' ', cleaned_jd)
    return cleaned_text


# clean emojis
def remove_emoji(string):
    emoji_pattern = re.compile("["
                        u"U0001F600-U0001F64F"  # emoticons
                        u"U0001F300-U0001F5FF"  # symbols & pictographs
                        u"U0001F680-U0001F6FF"  # transport & map symbols
                        u"U0001F1E0-U0001F1FF"  # flags (iOS)
                        u"U00002702-U000027B0"
                        u"U000024C2-U0001F251"
                        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# return cleaned dataframe
def cleaning(raw_data, source):
    if source == "indeed":
        data_temp = raw_data[['Job Title', 'Job Description']].copy()
        data_temp.rename(columns={"Job Title": "title", "Job Description": "description"}, inplace= True)

    elif source == "linkedin" or source == "career_builder" or source == "glassdoor" :
        data_temp = raw_data[['title','description']].copy()


    # Create source column
    data_temp['source'] = source
    print(source)

    # drop NA
    print('1 - Drop NA')
    data_temp.dropna(inplace= True)

    # reduce dataframe to title, description and source
    print('2 - Reduce')
    clean_data = data_temp[['title','description','source']]

    # lower case all strings
    print('3 - Lower')
    clean_data['description'] = clean_data['description'].apply(lambda x: x.lower())

    # apply clean_code function
    print('4 - Clean')
    clean_data['description'] = clean_data['description'].apply(clean_code)

    # remove emoji
    print('5 - Remove emoji')
    clean_data['description'] = clean_data['description'].apply(lambda x: remove_emoji(x))

    # unscape texts in html
    print('6 - Unscape HTML')
    clean_data['description'] = clean_data['description'].apply(lambda x: html.unescape(x))

    # remove punctuation
    print('7 - Punctuation')
    PUNCT_TO_REMOVE = string.punctuation
    clean_data['description'] = clean_data['description'].apply(lambda x: x.replace(PUNCT_TO_REMOVE, ''))

    #Remove digits
    print('8 - Remove digits')
    clean_data['description'] = clean_data['description'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))

    # Last drop NA
    print('9 - Last Drop NA')
    clean_data.dropna(inplace=True)

    # Reset index
    print('10 - reset index')
    clean_data = clean_data.reset_index(drop=True)

    print(f"{clean_data['description'].isna().value_counts()}")
    print(f"{clean_data['source'].isna().value_counts()}")
    print('DONE')
    return clean_data

def concat_df(df_list) :
    print("Start concat")
    final_df = pd.concat(df_list, ignore_index=True)
    final_df = final_df.reset_index(drop=True)

    final_df = final_df.drop_duplicates()
    final_df.dropna(inplace=True)

    print(f"{final_df['description'].isna().value_counts()}")
    print(f"{final_df['source'].isna().value_counts()}")

    return final_df

# def to_csv_1_50_all(df):
#     final_df_1k = df.sample(n=1000, random_state=42)
#     final_df_50k = df.sample(n=50000, random_state=42)
#     print("Create final CSV (1k, 50k, all)")

#     final_df_1k.to_csv('../raw_data/final_dataset_1k.csv', index=False)
#     final_df_50k.to_csv('../raw_data/final_dataset_50k.csv', index=False)
#     df.to_csv('../raw_data/final_dataset_all.csv', index=False)
#     print('CSVs created')



# reset index
def reset_df_index(df):
    df = df.reset_index(drop=True)
    return df

# drop na après concat
def drop_na(df):
    df = df.dropna(inplace=True)
    return df

# count nan après concat
def count_na(df):
    print(f"{df['description'].isna().value_counts()}")
    print(f"{df['source'].isna().value_counts()}")


if __name__ == '__main__':
    indeed = pd.read_csv('../../marketing_sample_for_trulia_com-real_estate__20190901_20191031__30k_data.csv')
    linkedin = pd.read_csv('../../job_postings.csv')
    career_builder = pd.read_json('../../career_builder_jobs_10501.json')
    glassdoor = pd.read_csv('../../glassdoor_en.csv', engine='python')

    concat_indeed_linkedin = concat_df([cleaning(indeed, 'indeed').reset_index(drop=True), cleaning(linkedin, 'linkedin').reset_index(drop=True)])
    drop_na(concat_indeed_linkedin)
    count_na(concat_indeed_linkedin)
    reset_df_index(concat_indeed_linkedin)
    print('concat Linkedin et Indeed')
    count_na(concat_indeed_linkedin)

    concat_indeed_linkedin_careerbuilder = concat_df([concat_indeed_linkedin.reset_index(drop=True), cleaning(career_builder, 'career_builder').reset_index(drop=True)])
    drop_na(concat_indeed_linkedin_careerbuilder)
    count_na(concat_indeed_linkedin_careerbuilder)
    reset_df_index(concat_indeed_linkedin_careerbuilder)
    print('concat ajout career builder')
    count_na(concat_indeed_linkedin_careerbuilder)


    concat_indeed_linkedin_careerbuilder_glassdoor = concat_df([concat_indeed_linkedin_careerbuilder.reset_index(drop=True), cleaning(glassdoor, 'glassdoor').reset_index(drop=True)])
    drop_na(concat_indeed_linkedin_careerbuilder_glassdoor)
    count_na(concat_indeed_linkedin_careerbuilder_glassdoor)
    reset_df_index(concat_indeed_linkedin_careerbuilder_glassdoor)
    print('concat ajout glassdoor')
    count_na(concat_indeed_linkedin_careerbuilder_glassdoor)

    # to_csv_1_50_all(concat_indeed_linkedin_careerbuilder_glassdoor)
