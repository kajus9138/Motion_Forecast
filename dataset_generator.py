import pandas as pd
import json


caminho_forecast_ambiental = r'C:\Users\ksilva\Documents\Motion_Forecast\dados_brutos/forecast_waves_20240127_20241031.json'
caminho_motion = r'C:\Users\ksilva\Documents\Motion_Forecast\dados_brutos/forecast_motion_20240127_20241031.json'

# Carregando dados ambientais***********************************
with open(caminho_forecast_ambiental, 'r') as file:
    data = json.load(file)

forecast_wave_data = data["forecastWave"]
df_forecast_wave = pd.DataFrame(forecast_wave_data)
df_forecast_wave = df_forecast_wave[['timestamp', 'wave_height', 'wave_height_max', 'peak_period', 'wave_height_swell',
                                    'wave_height_sea','wave_dir_swell', 'wave_dir_sea','peak_period_swell', 'peak_period_sea'
                                    ]]
df_forecast_wave['timestamp'] = pd.to_datetime(df_forecast_wave['timestamp'])
df_forecast_wave.set_index('timestamp', inplace=True)

# Carregando dados de movimento**********************************
with open(caminho_motion, 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)
df_motion = df[['timestamp','amp_roll', 'amp_pitch', 'amp_heave']]
df_motion['timestamp'] = pd.to_datetime(df_motion['timestamp'])
df_motion.set_index('timestamp', inplace=True)

df_motion.rename(columns={'amp_roll': 'roll', 'amp_pitch': 'pitch', 'amp_heave':'heave'}, inplace=True)
df_motion_3h = df_motion.resample('3H').median()

df_motion_3h.index.name = 'timestamp'
df_total = pd.merge(df_motion_3h, df_forecast_wave, left_index=True, right_index=True, how='inner')
#df_total.drop(['wave_height', 'wave_height_max', 'peak_period'], inplace=True, axis=1)

df_total.to_csv('df_envmotion.csv')


