import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yaml

def create_sequences_2(data, timestamps, n_steps, extra_features):
    X, y = [], []
   
    for i in range(len(data) - 2 * n_steps):
        current_time = timestamps[i]
        next_time = timestamps[i + 1]
        time_diff = (next_time - current_time).total_seconds() / 3600  # em horas

        if time_diff > 3:
            continue # caso a diferença entre as linhas consecutivas for maior que 3h a sequencia é pulada

        X_seq = data[i:i + n_steps]
        y_seq = data[i + n_steps:i + 2 * n_steps, :3]
        extra_seq = extra_features[i + n_steps:i + 2 * n_steps]

        X.append(np.hstack([X_seq, extra_seq]))
        y.append(y_seq)

    return np.array(X), np.array(y)


# Carregamento dos dados****************************************
with open("config.yml", 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

df_total = pd.read_csv(config['caminho_dataset'])
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

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, X_train.shape[2])))  # Todas as 12 variáveis são usadas como entrada
model.add(Dense(4 * 3))  # Ajuste a camada de saída para prever apenas as 3 variáveis-alvo em 4 passos de tempo
model.add(Reshape((4, 3)))  # Redimensione a saída para (4, 3), ou seja, 4 passos de tempo para 3 variáveis
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train[:, :,], epochs=30, batch_size=3, verbose=1)

print("modelo treinado")
