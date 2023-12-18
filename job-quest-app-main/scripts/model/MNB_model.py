from sklearn.naive_bayes import MultinomialNB

""" SCRIPT MODEL ML : """

def init_model():
    """ instantiate Multinomial Naive Bayes model """
    model = MultinomialNB(alpha=0.1)
    return model


def fit_model(model, X_train_transformed, y_train_transformed):
    model = model.fit(X_train_transformed, y_train_transformed)
    return model


def score_model(model, X_test_transformed, y_test_transformed):
    return model.score(X_test_transformed, y_test_transformed)


def make_prediction(model, X_test_transformed, label_encoder):
    y_pred = model.predict(X_test_transformed)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    return y_pred_decoded
