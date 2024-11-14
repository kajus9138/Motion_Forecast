import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yaml
import mlflow
import mlflow.keras

mlflow.set_experiment('LSTM_Motion_01')

# Função para criar sequências (já existente)
def create_sequences_2(data, timestamps, n_steps, extra_features):
    X, y = [], []
   
    for i in range(len(data) - 2 * n_steps):
        current_time = timestamps[i]
        next_time = timestamps[i + 1]
        time_diff = (next_time - current_time).total_seconds() / 3600  # em horas

        if time_diff > 3:
            continue  # caso a diferença entre as linhas consecutivas for maior que 3h a sequencia é pulada

        X_seq = data[i:i + n_steps]
        y_seq = data[i + n_steps:i + 2 * n_steps, :3]
        extra_seq = extra_features[i + n_steps:i + 2 * n_steps]

        X.append(np.hstack([X_seq, extra_seq]))
        y.append(y_seq)

    return np.array(X), np.array(y)

# Carregamento dos dados****************************************
with open("config.yml", 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

df_total = pd.read_csv(config['dataset_path'])
df_total.set_index('timestamp', inplace=True)
df_total.dropna(inplace=True)

# Processamento para treinamento********************************
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_total[['heave', 'roll', 'pitch', 'wave_height', 'wave_height_max',
                                             'peak_period', 'wave_height_swell', 'wave_height_sea',
                                             'wave_dir_swell', 'wave_dir_sea', 'peak_period_swell', 'peak_period_sea']])

n_steps = 4
fim = 1960
timestamps = df_total.index[:fim] 
timestamps = pd.to_datetime(df_total.index[:fim])

X_features = scaled_data[:fim, :]
extra_features = scaled_data[:fim + 1, 3:]  # Extra_features será um passo adiantado em relação a X_features

X_train, y_train = create_sequences_2(X_features, timestamps, n_steps, extra_features)

timestamps_test = df_total.index[fim:]
timestamps_test = pd.to_datetime(df_total.index[fim:])
X_features_test = scaled_data[fim:, :]
extra_features_test = scaled_data[fim + 1:, 3:]  
X_test, y_test = create_sequences_2(X_features_test, timestamps_test, n_steps, extra_features_test)

# Início do rastreamento com MLflow
with mlflow.start_run():

    # Definindo parâmetros a serem rastreados
    mlflow.log_param("dataset_path", config['dataset_path'])
    mlflow.log_param("n_steps", config['n_steps'])
    mlflow.log_param("fim", config['fim'])
    mlflow.log_param("epochs", config['epochs'])
    mlflow.log_param("batch_size", config['batch_size'])
    mlflow.log_param("units", config['units'])
    mlflow.log_param("recurrent_activation", config["recurrent_activation"])
    mlflow.log_param("activation", config["activation"])
    mlflow.log_param("optimizer", config['optimizer'])
    

    # Definindo o modelo
    model = Sequential()
    model.add(LSTM(units=config['units'], activation=config["activation"], input_shape=(config['n_steps'], X_train.shape[2])))  
    model.add(Dense(4 * 3))  
    model.add(Reshape((4, 3)))  
    model.compile(optimizer=config['optimizer'], loss='mse')

    # Treinando o modelo
    model.fit(X_train, y_train[:, :,], epochs=config["epochs"], batch_size=config["batch_size"], verbose=1)

    # Logando o modelo no MLflow
    mlflow.keras.log_model(model, "model")

    # Calculando e logando as métricas
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test.reshape(-1, 3), y_pred.reshape(-1, 3))
    mae = mean_absolute_error(y_test.reshape(-1, 3), y_pred.reshape(-1, 3))

    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("MAE", mae)

    print("Modelo treinado e rastreado no MLflow")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")

