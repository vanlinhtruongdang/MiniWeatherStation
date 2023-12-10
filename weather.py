import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import schedule
import time
import os
from SoftOrdering1DCNN import SoftOrdering1DCNN
import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

API_Key = "c8247668d0e5640ca269aa61eabfc4e3"
Dataset = pd.DataFrame(columns=['datetime', 'temperature', 'wind_speed', 'humidity', 'cloud', 'pressure', 'visibility', 'rain', 'predict'])

Latitude = 10.869778736885038
Longtitude = 106.80280655508835
Timestamp = None
pipeline = None

OpenweatherAPI = f"https://api.openweathermap.org/data/2.5/weather"
Params = {
        'lat': Latitude,
        'lon': Longtitude,
        'appid': API_Key,
        'units': 'metric',
    }

def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

def PipelineInit():
    global pipeline
    from sklearn.preprocessing import StandardScaler, Normalizer
    from sklearn.pipeline import Pipeline

    TrainData = np.load('./PreprocessedData/Data.npy')
    TrainLabel = np.load('./PreprocessedData/Target.npy')
    pipeline = Pipeline([
        ('Standard scaling', StandardScaler()),
        ('Normalize', Normalizer())]).fit(X=TrainData, y=TrainLabel)
    
def ConvertUTCTimestamp(UTCTimestamp, getdatetime = False):
    UTC_Datetime = datetime.utcfromtimestamp(UTCTimestamp)
    HCM_Timezone = timezone(timedelta(hours=7))
    VN_Datetime = UTC_Datetime.replace(tzinfo=timezone.utc).astimezone(HCM_Timezone)

    if not getdatetime:
        VN_Datetime = VN_Datetime.strftime('%Y')
    else:
        VN_Datetime = VN_Datetime.strftime('%H:%M:%S %d-%m-%Y')

    return VN_Datetime

def SaveDataset(Timestamp):
    global Dataset

    Datetime = ConvertUTCTimestamp(Timestamp)
    FileName = f'{Datetime}.csv'
    SaveDirectory = f'/usr/share/grafana/public/WeatherData'

    FilePath = os.path.join(SaveDirectory, FileName)
    Dataset.drop_duplicates(inplace=True)
    Dataset.to_csv(FilePath, index=False)
    prGreen(f"Dataset was saved successfully: {FilePath}")


def GetWeatherData():
    global Dataset
    global Timestamp

    session = requests.session()
    response = session.get(url=OpenweatherAPI, params=Params)

    Model = SoftOrdering1DCNN(5, 2)
    Model.load_state_dict(torch.load("./ModelCheckpoint/SoftOrdering1DCNN.pth"))
    Model.eval().to(device)

    if response.status_code == 200:
        prGreen("Get weather data from OpenWeather successfully!")
        ResponseData = response.json()
        Timestamp = ResponseData['dt']

        CityName = ResponseData['name']
        lon = str(ResponseData['coord']['lon'])
        lat = str(ResponseData['coord']['lat'])
        WeatherState = str(ResponseData['weather'][0]['description'])

        print(f'Location: {CityName} ({lon}, {lat})')
        print(f'Weather state: {WeatherState}')

        WeatherData = {
            'datetime': ConvertUTCTimestamp(Timestamp, getdatetime=True),
            'temperature': float(ResponseData['main']['temp']), # Celius
            'wind_speed': float(ResponseData['wind']['speed']), # metter/s
            'humidity': float(ResponseData['main']['humidity']), # Percentage
            'cloud': float(ResponseData['clouds']['all']), # Percentage
            'pressure': float(ResponseData['main']['pressure']), # hpa
            'visibility': float(ResponseData['visibility']), # metter
            'rain': None
        }

        Dataset = Dataset.append(WeatherData, ignore_index=True)

        Features = ['temperature', 'wind_speed', 'humidity', 'cloud', 'pressure']
        DataNumpy = Dataset[Features].iloc[-1].values
        DataNumpy = pipeline.transform(DataNumpy.reshape(1,-1))
        
        Data = torch.from_numpy(DataNumpy).float().to(device)
        Predict = Model(Data)
        print(Predict)

        Rain = float(Predict[0][1]) * 100
        Dataset.at[Dataset.index[-1], 'rain'] = Rain

        print(Dataset.iloc[-1])
        SaveDataset(Timestamp)

    else:
        prRed("An error occurred while retrieving weather data from OpenWeather!")


if __name__ == '__main__':
    PipelineInit()
    
    schedule.every(interval=3).seconds.do(GetWeatherData)

    while True:
        schedule.run_pending()
        time.sleep(1)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import schedule
import time
import os
from SoftOrdering1DCNN import SoftOrdering1DCNN
import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

API_Key = "c8247668d0e5640ca269aa61eabfc4e3"
Dataset = pd.DataFrame(columns=['datetime', 'temperature', 'wind_speed', 'humidity', 'cloud', 'pressure', 'visibility', 'rain', 'predict'])

Latitude = 10.869778736885038
Longtitude = 106.80280655508835
Timestamp = None
pipeline = None

OpenweatherAPI = f"https://api.openweathermap.org/data/2.5/weather"
Params = {
        'lat': Latitude,
        'lon': Longtitude,
        'appid': API_Key,
        'units': 'metric',
    }

def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

def PipelineInit():
    global pipeline
    from sklearn.preprocessing import StandardScaler, Normalizer
    from sklearn.pipeline import Pipeline

    TrainData = np.load('./PreprocessedData/Data.npy')
    TrainLabel = np.load('./PreprocessedData/Target.npy')
    pipeline = Pipeline([
        ('Standard scaling', StandardScaler()),
        ('Normalize', Normalizer())]).fit(X=TrainData, y=TrainLabel)
    
def ConvertUTCTimestamp(UTCTimestamp, getdatetime = False):
    UTC_Datetime = datetime.utcfromtimestamp(UTCTimestamp)
    HCM_Timezone = timezone(timedelta(hours=7))
    VN_Datetime = UTC_Datetime.replace(tzinfo=timezone.utc).astimezone(HCM_Timezone)

    if not getdatetime:
        VN_Datetime = VN_Datetime.strftime('%Y')
    else:
        VN_Datetime = VN_Datetime.strftime('%H:%M:%S %d-%m-%Y')

    return VN_Datetime

def SaveDataset(Timestamp):
    global Dataset

    Datetime = ConvertUTCTimestamp(Timestamp)
    FileName = f'{Datetime}.csv'
    SaveDirectory = f'/usr/share/grafana/public/WeatherData'

    FilePath = os.path.join(SaveDirectory, FileName)
    Dataset.drop_duplicates(inplace=True)
    Dataset.to_csv(FilePath, index=False)
    prGreen(f"Dataset was saved successfully: {FilePath}")


def GetWeatherData():
    global Dataset
    global Timestamp

    session = requests.session()
    response = session.get(url=OpenweatherAPI, params=Params)

    Model = SoftOrdering1DCNN(5, 2)
    Model.load_state_dict(torch.load("./ModelCheckpoint/SoftOrdering1DCNN.pth"))
    Model.eval().to(device)

    if response.status_code == 200:
        prGreen("Get weather data from OpenWeather successfully!")
        ResponseData = response.json()
        Timestamp = ResponseData['dt']

        CityName = ResponseData['name']
        lon = str(ResponseData['coord']['lon'])
        lat = str(ResponseData['coord']['lat'])
        WeatherState = str(ResponseData['weather'][0]['description'])

        print(f'Location: {CityName} ({lon}, {lat})')
        print(f'Weather state: {WeatherState}')

        WeatherData = {
            'datetime': ConvertUTCTimestamp(Timestamp, getdatetime=True),
            'temperature': float(ResponseData['main']['temp']), # Celius
            'wind_speed': float(ResponseData['wind']['speed']), # metter/s
            'humidity': float(ResponseData['main']['humidity']), # Percentage
            'cloud': float(ResponseData['clouds']['all']), # Percentage
            'pressure': float(ResponseData['main']['pressure']), # hpa
            'visibility': float(ResponseData['visibility']), # metter
            'rain': None
        }

        Dataset = Dataset.append(WeatherData, ignore_index=True)

        Features = ['temperature', 'wind_speed', 'humidity', 'cloud', 'pressure']
        DataNumpy = Dataset[Features].iloc[-1].values
        DataNumpy = pipeline.transform(DataNumpy.reshape(1,-1))
        
        Data = torch.from_numpy(DataNumpy).float().to(device)
        Predict = Model(Data)
        print(Predict)

        Rain = float(Predict[0][1]) * 100
        Dataset.at[Dataset.index[-1], 'rain'] = Rain

        print(Dataset.iloc[-1])
        SaveDataset(Timestamp)

    else:
        prRed("An error occurred while retrieving weather data from OpenWeather!")


if __name__ == '__main__':
    PipelineInit()
    
    schedule.every(interval=3).seconds.do(GetWeatherData)

    while True:
        schedule.run_pending()
        time.sleep(1)
