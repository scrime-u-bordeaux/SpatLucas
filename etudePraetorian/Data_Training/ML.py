DATASET_FOLDER = "resampled_results/"
RESULT_FOLDER = "Results/"
RESULT_FOLDER_PREDICT = "Results/Predict_coords/"
RESULT_FOLDER_MAX = "Results/MaxData/"

import pandas as pd
import os 
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import inspect


"""
Load all datasets from the DATASET_FOLDER
   returns a tuple of (df_train, df_test)
   """
def load_all_datasets(folder):
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    csv_files = sorted(csv_files)

    test_file = csv_files[-1]  # Last track is the test set
    # All other files are used for training
    train_files = csv_files[:-1]

    train_dfs = []
    for i, f in enumerate(train_files):
        df = pd.read_csv(os.path.join(folder, f))
        df["track_id"] = i
        train_dfs.append(df)

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test = pd.read_csv(os.path.join(folder, test_file))
    df_test["track_id"] = len(train_dfs)

    return df_train, df_test


def main():
    # Load all training and test datasets
    df_train, df_test = load_all_datasets(DATASET_FOLDER)

    label_encoder = LabelEncoder()
    df_train['region_encoded'] = label_encoder.fit_transform(df_train['region'])
    df_test['region_encoded'] = label_encoder.transform(df_test['region'])

    features = ['rms', 'region_encoded', 'temps', 'mesures']
    X_train = df_train[features]
    y_train = df_train[['x', 'y']]
    X_test = df_test[features]
    y_test = df_test[['x', 'y']]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose a model
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    
    print(model.__class__)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

    print("=== Évaluation modèle simple ===")
    print(f"MSE : {mse:.4f}")
    print(f"R2  : {r2:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"MAPE : {mape:.4f}")

    from sklearn.dummy import DummyRegressor
    

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train_scaled, y_train)
    y_dummy = dummy.predict(X_test_scaled)

    mse_dummy = mean_squared_error(y_test, y_dummy)
    print("MSE Dummy (baseline) :", mse_dummy)
    write_coord_in_file(y_pred)


"""
Write the predicted coordinates to a CSV file.
Parameters:
  y_pred: The predicted coordinates as a DataFrame or 2D array.
  filename: The name of the output CSV file (default is "predicted_coordinates.csv").
"""
def write_coord_in_file(y_pred, filename="predicted_coordinates.csv"):
    """
    Écrit uniquement les coordonnées prédites x et y dans un fichier CSV.
    """

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    df_coords = pd.DataFrame(y_pred, columns=["x_pred", "y_pred"])

    frame = inspect.currentframe()
    while frame:
        if 'model' in frame.f_locals:
            model_class_name = frame.f_locals['model'].__class__.__name__
            break
        frame = frame.f_back
    else:
        model_class_name = "UnknownModel"

    filename_with_model = os.path.join("Predict_coords/", f"{model_class_name}_{filename}")
    output_path = os.path.join(RESULT_FOLDER, filename_with_model)
    df_coords.to_csv(output_path, index=False)

    print(f"Coordonnées prédictes écrites dans : {output_path}")
    
"""
Convert the predicted coordinates to a format compatible with Max/MSP.
Parameters:
  csv_file: The path to the CSV file containing the predicted coordinates.
    """
def convert_to_max_data(csv_file):
    df = pd.read_csv(csv_file)

    output_lines = []
    n_frames = len(df)
    index = 1  # starting with 1 because 0 is the name of the track

    output_lines.append("0, id 09-Temps-Mort;")

    last_x, last_y = df.iloc[0]['x_pred'], df.iloc[0]['y_pred']
    norm_time = 0.0
    output_lines.append(f"{index}, {norm_time:.6f} {last_x:.6f} {last_y:.6f};")
    index += 1

    for i in range(1, n_frames):
        x, y = df.iloc[i]['x_pred'], df.iloc[i]['y_pred']
        norm_time = i / (n_frames - 1) if n_frames > 1 else 0.0  # Normalisé entre 0 et 1
        if x != last_x or y != last_y:
            output_lines.append(f"{index}, {norm_time:.6f} {x:.6f} {y:.6f};")
            last_x, last_y = x, y
            index += 1

    os.makedirs(RESULT_FOLDER_MAX, exist_ok=True)
    with open(os.path.join(RESULT_FOLDER_MAX, "seq3.txt"), "w") as f:
        for line in output_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()

    # Convert the predicted coordinates to Max/MSP format

    # csv_file = os.path.join(RESULT_FOLDER_PREDICT, "LinearRegression_predicted_coordinates.csv")
    # convert_to_max_data(csv_file)