import pickle
from fastapi import FastAPI
from scripts.model.registry import load_model, load_category
from scripts.preprocessor.preprocessing_query import preproc_query

app = FastAPI()
app.state.model = load_model()
app.state.model_conv = load_model(type="conv")
app.state.category = load_category()

# load model
model_filename = "./outputs/models_ml/model.pickle"
model = pickle.load(open(model_filename, "rb"))

# load vectorizer
vectorizer_filename = "./outputs/vectorizers_ml/ml_vectorizer.pickle"
vectorizer = pickle.load(open(vectorizer_filename, "rb"))

# load label encoder
labelencoder_filename = "./outputs/labelencoders_ml/ml_labelencoder.pickle"
label_encoder = pickle.load(open(labelencoder_filename, "rb"))

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/ping")
def pong():
    return {"ping": "pong!"}

@app.post("/predict")
def post_job_list_dense(type, query):

    #Get the model or assert it
    if type == "conv":
        model = app.state.model_conv
        assert model is not None
    else:
        model = app.state.model
        assert model is not None

    #Embed query
    query_outputs = preproc_query(query)

    #Get result from the model
    result = model.predict(query_outputs)

    #Get the cat of your one hot encoding
    cat = app.state.category

    #Prepare a nice dict for the answer and filter with result sup 0.9

    paired = zip(cat[0].tolist(), result.tolist()[0])
    filtered_pairs = [(c, r) for c, r in paired if float(r) > 0.1]
    sorted_paired = sorted(filtered_pairs, key=lambda x: x[1], reverse= True)
    if len(sorted_paired) > 4:
        result_sorted = sorted_paired[:4]
    if len(sorted_paired) <= 4:
        result_sorted = sorted_paired
    result = [(f"You would be a phenomenal {c.capitalize()}", f"Matching is {int(round(r,2)*100)}%") for c, r in result_sorted]
    result = {
        'top_match': {
            'id': 'swe',
            'title': '',
            'matching_score': ''
        },
        'matches': [
            {
                'id':
                'title': ''
                'matching_score': ''
            },
            {
                'title': ''
                'matching_score': ''
            },
            {
                'title': ''
                'matching_score': ''
            },
            {
                'title': ''
                'matching_score': ''
            },
        ]
    }

    my_dict = dict(result)

    return my_dict

# predict
@app.post("/predict_ml")
def post_category(query):
    clean_query = vectorizer.transform([query])
    print(clean_query)
    y_pred = model.predict(clean_query)
    result = label_encoder.inverse_transform(y_pred)
    print(result[0])

    return result[0]
