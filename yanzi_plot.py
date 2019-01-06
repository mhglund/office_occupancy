from datetime import datetime
import pandas as pd
import numpy as np
import arrow
from dateutil import tz

def get_csv_data(filename):
    # Läs in csv-fil
    df = pd.read_csv(filename, sep=';', encoding = "utf-8")
    # Ta bort mellanslag
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

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

# Tvinga på rätt tidszon (för att kunna konvertera alla datum till datetime)
def convert_timeformat(text):
    temp = convert_time(text)
    temp.tzinfo = tz.gettz('GMT')
    return temp.datetime

def detect_activity(df):
    # Räkna ut hur mycket som ändrats för varje sensor sedan förra mätvärdet
    changes = df.sort_values('time').groupby('sensorId')['value'].diff()

    # pseudo: spara tidpunkten (och sensorId?) temp om värdet är true, för att kunna räkna
    # tidsdifferensen mellan den och nästa värde som är true (om samma sensor) –
    # "släng" (gör till false) den första om det är över fem minuter?

    # skriva nån felhantering för när sensorn nollställer

    # returnera true/false beroende på om ändringen är större än 0 / lika med 1?
    return changes.transform(lambda x: x == 1)

def add_rooms(df):
    # Kombinera sensortyp (typ av plats) med aktivitet på ett
    # lite effektivare sätt än nästlade loopar...
    sensorlist = get_csv_data('sensor_list.csv')
    return df.join(sensorlist.set_index('Sensor'), on='sensorId')

# Läs in filen
activity = get_csv_data('yanzi_motion_20000l_slump.csv')

# parsea tidsformaten
activity['time'] = pd.to_datetime(
    activity['time'].transform(convert_timeformat)
)

# Plocka ut alla rader i vissa kolumner
activity = activity.loc[:, ['time', 'sensorId', 'value', 'movement']]

# Sätt true/false beroende på aktivitet
#activity['movement'] = detect_activity(activity)

# Koppla aktivitet till rumstyp
activity = add_rooms(activity)

# Extrahera timme och veckodag ur datumen
activity['hour'] = activity.apply(lambda row: row['time'].hour, axis=1)
activity['weekday'] = activity.apply(lambda row: row['time'].dayofweek, axis=1)

print('-------------------------- 25 slumpade --------------------------')
print(activity.sample(25).sort_values(by=['sensorId', 'time']))

activity_mitt = activity.head(7500)
en_sensor = activity.loc[activity['sensorId'] == 'EUI64-0080E1030004F23D-4-Motion']

print('-------------------------- en sensor --------------------------')
print(en_sensor.tail(500).sort_values(by=['sensorId', 'time']))
#en_sensor.to_csv('sensor0080E1030004F23D.csv', sep='\t', encoding='utf-8')

# Plocka ut rader med tomma värden
tomma = activity.isnull().sum().sum()

print('SAKNAR VÄRDE (antal):')
print(tomma)
print('--------------------------')

#print('-------------------------- 250 mitten --------------------------')
#print(activity_mitt.tail(500).sort_values(by=['sensorId', 'time']))


from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# Ta fram medelvärde för de som har samma hour och weekday
average_movement = activity.groupby(['hour', 'weekday'], as_index=False).sum()
#Delat på 1,5 för att inte prickarna ska bli för stora
average_movement['movement'] = average_movement['movement']/1.5

output_file('scatter.html')
p = figure(
    title='Aktivitet vid timma och veckodag',
    x_axis_label='Veckodag',
    y_axis_label='Timma',
    x_range=(-1, 7),
    y_range=(-1, 25),
)
p.circle(
    source=ColumnDataSource(data=average_movement),
    x='weekday',
    y='hour',
    size='movement',
    color='black',
    alpha=0.15
)

# Öppna webbläsare och visa plot, bortkommenterad för enkelhetens skull
show(p)
