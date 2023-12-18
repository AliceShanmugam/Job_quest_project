import glob
import os
import time
from tensorflow import keras
from scripts.params import *

def save_model(model: keras.Model = None, type: str = 'dense') -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    """

    # Save model locally
    if type == "conv":
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "models_conv", f"model.h5")
        model.save(model_path)
    else:
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "models_dense", f"model.h5")
        model.save(model_path)

    print("✅ Model saved locally")

    return None

def load_model(type: str = 'dense') -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    Return None (but do not Raise) if no model is found
    """

    if MODEL_TARGET == "local":
        print(f"\n\nLoad latest model from local registry...")

        # Get the latest model version name by the timestamp on disk
        if type == "conv":
            local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models_conv")
        else:
            local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models_dense")

        print(f'\n\nLOAD MODEL FROM {local_model_directory}')
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(f"\nLoad latest model from disk...")
        print(f'\n\nMOST RECENT MODEL: {most_recent_model_path_on_disk}')

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("\n\n✅ Model loaded from local disk")

        return latest_model

    else:
        print('NO MODEL LOADED')
        return None

def save_category(category) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    category_path = os.path.join(LOCAL_REGISTRY_PATH, "categories", f"category_{timestamp}.npy")
    np.save(category_path, category)

    print("✅ Category encoded saved locally")

    return None

def load_category():
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    Return None (but do not Raise) if no model is found
    """

    if MODEL_TARGET == "local":
        print(f"\nLoad latest category from local registry...")

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "categories")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(f"\nLoad latest category from disk...")

        latest_category = np.load(most_recent_model_path_on_disk, allow_pickle=True)

        print("✅ Category loaded from local disk")

        return latest_category

    else:
        return None
