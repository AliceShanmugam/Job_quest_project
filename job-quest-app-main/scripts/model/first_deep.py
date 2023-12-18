from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

def init_model(X,y, type):
    """
    Init a dense model
    type = dense or conv
    """
    if type == "dense":
        print("Dense Model")
        dense_model = Sequential([
        Dense(512, activation='relu', input_shape=(X.shape[1],X.shape[2])),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Flatten(),
        Dense(y.shape[1], activation='softmax')
        ])
        print("Compile Model")
        dense_model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        return dense_model

    if type == "conv":
        print("Conv1D Model")
        conv_model = Sequential([
        Dense(512, activation='relu', input_shape=(512,128)),
        Conv1D(50, kernel_size=5),
        Conv1D(30, kernel_size=5),
        Dense(128, activation='relu'),
        Flatten(),
        Dense(10, activation='sigmoid')
        ])
        print("Compile Model")
        conv_model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

        return conv_model

def fit_model(model, X, y, batch_size, patience, validation_data):
    es = EarlyStopping(patience=patience, restore_best_weights=True)
    print("Fitting Model")
    history = model.fit(X, y,
        batch_size=batch_size,
        epochs=10,
        validation_split=0.3,
        validation_data=validation_data,
        callbacks=[es])
    print("Model Fitted")
    return model, history

def evaluate_model(model, X, y):
    return model.evaluate(X, y)

def predict_model(model, X):
    return model.predict(X)

def cross_val_deep(X, y, type):
    """
    Cross val a deep model
    """

    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    results = []

    for train_index, test_index in kf.split(X):
        # Split the data into train and test

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize the model
        model = init_model(X_train,y_train, type)

        # Fit the model on the training data
        model_fitted, history = fit_model(model, X_train, y_train)

        print(history)
        # Evaluate the model on the testing data
        res = evaluate_model(model_fitted, X_test, y_test)
        results.append(res)

    return results
