from datetime import datetime
import pandas as pd
import numpy as np
import arrow
from dateutil import tz


# Läs in filen
df = pd.read_csv('yanzi_1000l.csv', sep=';', encoding = "utf-8")
sensorlist = pd.read_csv('sensor_list.csv', sep=';', encoding = "utf-8")

# Plocka ut 10 första raderna för att inte skriva ut så mkt
df_top = df.head(10)

# Lite utskrifter
print('TOPP 10:')
print(df_top)
print('--------------------------')
print('BESKRIVNING:')
print(df.describe())
print('--------------------------')
print('INFO:')
print(df.info())
print('--------------------------')

# Ta bort alla whitespaces
df_trimmed = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
sensorlist_trimmed = sensorlist.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


# Konvertera datumformatet
def convert_time(text):
    try:
        return arrow.get(text)
    except:
        try:
             return arrow.get(text, 'YYYY-M-D HH:mm:ss')
        except:
            try:
               return arrow.get(text[:10]+'.'+text[-3:])
            except:
                return arrow.get(text[:-4], 'ddd, D MMMM YYYY HH:mm:ss')

def convert_timeformat(text):
    temp = convert_time(text)
    temp.tzinfo = tz.gettz('GMT')
    return temp.datetime

df_trimmed['time'] = pd.to_datetime(df_trimmed.loc[:,'time'].transform(convert_timeformat))

print('25 slumpade rader:')
print(df_trimmed.sample(25))
print('--------------------------')
print('INFO m. nytt datumformat:')
print(df_trimmed.info())
print('--------------------------')


# Plocka ut en viss sensor (rader med ett visst kolumnvärde)
en_sensor = df_trimmed.loc[df_trimmed['sensorId'] == 'EUI64-0080E10300045738-4-Motion']
# Plocka ut alla rader i vissa kolumner
df_vissa_kolumner = df_trimmed.loc[:, ['time', 'location', 'sensorId', 'value']]


# Räkna ut hur mycket som ändrats för varje sensor sedan förra mätvärdet
changed = df_vissa_kolumner.groupby('sensorId')['value'].diff()

# Konvertera till boolean
haschanged = changed.transform(lambda x: x > 0)

# Konvertera till ny DataFrame och slå ihop med gamla
df_movement = df_vissa_kolumner.join(pd.DataFrame({'movement': haschanged}))


#df_vissa_kolumner.join(pd.DataFrame({'movement': df_vissa_kolumner.groupby('sensorId')['value'].diff().transform(lambda x: x > 0)}))

print('Med movement (true/false):')
print(df_movement.head(100))
print('--------------------------')
print('SENSOR: EUI64-0080E10300045738-4-Motion')
print(en_sensor)
print('--------------------------')


# Plocka ut rader med tomma värden
tomma = df_trimmed.isnull().values.any()

print('SAKNAR EJ VÄRDE:')
print(tomma)
print('--------------------------')


# Skapar en ny DF(kopia) för att kunna lägga till rum
# newDF = pd.DataFrame(df_trimmed)

# def addRooms():
#     roomsArray = []
#     for indexYanzi, rowYanzi in df_trimmed.iterrows():
#         for index, row in sensorlist.iterrows():
#             if rowYanzi[4] == row[0]:
#                 roomsArray.append(row[1])

#     newDF['Room'] = pd.Series(roomsArray, index=newDF.index)

# addRooms()

def get_csv_data(filename):
    df = pd.read_csv(filename, sep=';', encoding = "utf-8")
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

def add_rooms(df):
    # Kombinera sensortyp med aktivitet på ett lite effektivare
    # sätt än nästlade loopar...
    sensorlist = get_csv_data('sensor_list.csv')
    return df.join(sensorlist.set_index('Sensor'), on='sensorId')

df_trimmed = add_rooms(df_trimmed)

print('VÄRDEN MED RUMINFORMATION (25 slumpvisa):')
print(df_trimmed.sample(25))
print('--------------------------')
