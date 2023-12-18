import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
MODEL_TARGET = os.environ.get("MODEL_TARGET")

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")

DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE")
DOCKER_MEMORY = os.environ.get("DOCKER_MEMORY")

##################  CONSTANTS  #####################
CUR_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(CUR_DIR)

LOCAL_DATA_PATH = os.path.join(PROJECT_DIR, "outputs", "data")
LOCAL_REGISTRY_PATH =  os.path.join(PROJECT_DIR, "outputs")

TARGET_JOB_TITLES = [ 'project manager', 'software engineer', 'business analyst', 'product manager', 'data scientist', 'data analyst', 'data engineer', 'business development representative','devops engineer', 'marketing manager', 'account manager', 'customer success manager','accountant', 'account executive', 'financial analyst', 'system engineer', 'qa engineer','machine learning engineer', 'general manager', 'sales manager', 'sales engineer','database administrator', 'administrative assistant', 'operations manager', 'network engineer','electrical engineer', 'technical support engineer', 'mechanical engineer', 'store manager','business development manager', 'business development representative','business development executive','sales development representative','software developer','software development engineer','it project manager','product owner','program manager','net developer','project management','customer service representative','big data engineer','technical project manager','project engineer','business intelligence analyst','sales representative','inside sales representative','outside sales representative','project coordinator','it business analyst','business systems analyst','java software engineer','backend engineer','product marketing manager','systems engineer','product specialist','sales associate','python developer','qa engineer','test engineer','developer','quality assurance engineer','full stack engineer','qa automation engineer','software test engineer','sales executive','application developer','solutions engineer','solutions architect','frontend engineer','product development manager','business system analyst','data business analyst']
FINAL_JOB_TITLES = ['project manager', 'software engineer', 'business analyst', 'product manager', 'data scientist', 'data analyst', 'data engineer', 'sales representative','devops engineer', 'marketing manager', 'customer success manager', 'account executive', 'quality engineer', 'solutions engineer', 'system engineer']

GROUPING_JOB_DICT = {
    'business development manager': 'sales representative',
    'business development representative': 'sales representative',
    'business development executive': 'sales representative',
    'sales development representative': 'sales representative',
    'business development executive': 'sales representative',
    'software developer': 'software engineer',
    'software development engineer': 'software engineer',
    'it project manager': 'project manager',
    'product owner': 'product manager',
    'program manager': 'project manager',
    'net developer': 'software engineer',
    'project management': 'project manager',
    'customer service representative': 'customer success manager',
    'big data engineer': 'data engineer',
    'technical project manager': 'project manager',
    'project engineer': 'project manager',
    'business intelligence analyst': 'data analyst',
    'sales representative': 'sales representative',
    'inside sales representative': 'sales representative',
    'outside sales representative': 'sales representative',
    'project coordinator': 'project manager',
    'it business analyst': 'business analyst',
    'business systems analyst': 'business analyst',
    'java software engineer': 'software engineer',
    'backend engineer': 'software engineer',
    'product marketing manager': 'marketing manager',
    'systems engineer': 'system engineer',
    'product specialist': 'product manager',
    'sales representative': 'sales representative',
    'sales associate': 'business development representative',
    'python developer': 'software engineer',
    'qa engineer': 'quality engineer',
    'test engineer': 'quality engineer',
    'developer': 'software engineer',
    'quality assurance engineer': 'quality engineer',
    'full stack engineer': 'software engineer',
    'qa automation engineer': 'quality engineer',
    'software test engineer': 'software engineer',
    'sales executive': 'account executive',
    'application developer': 'software engineer',
    'frontend engineer': 'software engineer',
    'product development manager': 'product manager',
    'business system analyst': 'business analyst',
    'data business analyst': 'data analyst',
    'solutions consultant': 'solutions engineer',
    'sales engineer': 'solutions engineer',
    'customer support manager': 'customer success manager',
    'account manager': 'customer success manager',
    'sales manager': 'account executive',
    'machine learning engineer':'data scientist'
}
