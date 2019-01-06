from collections import namedtuple
import functools
from bokeh.io import show, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
import bokeh.palettes as palettes
from bokeh.plotting import figure
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import neighbors

def get_csv_data(filename):
    "Läs in CSV-fil"
    return pd.read_csv(filename, sep=';', encoding = "utf-8")

def add_rooms(df):
    "Kombinera typ av plats för sensorn med aktivitet"
    sensorlist = get_csv_data('sensor_list.csv')
    return df.join(sensorlist.set_index('Sensor'), on='sensorId')

def prepare_data(activity):
    "Parsea tidsformat, skapa/plocka ut intressanta kolumner"
    activity = activity.loc[:, ['time', 'sensorId', 'value', 'movement']]

    # Koppla aktivitet till rumstyp
    activity = add_rooms(activity)

    # Skapa datetime-objekt i pandas
    activity['time'] = pd.to_datetime(activity['time'])

    # Lägger till kolumner för olika tidsenheter
    activity['minute'] = activity.apply(lambda row: row['time'].minute, axis=1)
    activity['hour'] = activity.apply(lambda row: row['time'].hour, axis=1)
    activity['dayminute'] = activity['hour']*60 + activity['minute']
    activity['day'] = activity.apply(lambda row: row['time'].day, axis=1)
    activity['weekday'] = activity.apply(lambda row: row['time'].dayofweek, axis=1)
    activity['yearweek'] = activity.apply(lambda row: row['time'].weekofyear, axis=1)
    activity['weekhour'] = activity['weekday']*24 + activity['hour']

    return activity

def print_some(activity):
    "Skriv ut slumpade observationer"
    print('--------------------------')
    print(activity.sample(15))
    print('--------------------------')
    return activity

def plot_data(activity):
    "Plotta datat på olika sätt"
    output_file('scatter.html')

    movement_dayminute = activity.groupby('dayminute').sum()
    dayminute = figure(
        title='Aktivitet per minut en snittdag',
        x_axis_label='Minut under dagen',
        y_axis_label='Aktivitet'
    )
    dayminute.line(
        source=ColumnDataSource(data=movement_dayminute),
        x='dayminute',
        y='movement',
        color='black'
    )

    movement_weekhour = activity.groupby(
        ['weekhour', 'hour', 'weekday'],
        as_index=False
    ).sum()
    week_vs_hour = figure(
        title='Aktivitet vid timme och veckodag',
        x_axis_label='Veckodag',
        y_axis_label='Timme',
        x_range=(-1, 7),
        y_range=(-1, 25),
    )
    week_vs_hour.circle(
        source=ColumnDataSource(data=movement_weekhour),
        x='weekday',
        y='hour',
        size='movement',
        color='black',
        alpha=0.15
    )

    weekhour = figure(
        title='Aktivitet per timme i veckan',
        x_axis_label='Timme i veckan',
        y_axis_label='Aktivitet',
        x_range=(-12, 24*7.5)
    )
    weekhour.line(
        source=ColumnDataSource(data=movement_weekhour),
        x='weekhour',
        y='movement',
        color='black'
    )

    movement_yearweek = activity.groupby(
        ['yearweek', 'Type'], as_index=False
    ).sum()
    yearweek = figure(
        title='Aktivitet per vecka om året (rumstyper)',
        x_axis_label='Vecka på året',
        y_axis_label='Aktivitet'
    )
    for colour, room in enumerate(set(movement_yearweek['Type'])):
        yearweek.line(
            source=ColumnDataSource(
                data=movement_yearweek.loc[movement_yearweek['Type'] == room]
            ),
            x = 'yearweek',
            y = 'movement',
            legend = 'Type: ' + room,
            color = palettes.Category10_5[colour]
        )

    show(column(week_vs_hour, weekhour, yearweek, dayminute))

    return activity

def correlations(df, over, col, val):
    "Korrelera kolumnen 'col' med hänsyn till kolumnen 'val' aggregerat över 'over' i 'df'"
    df = df.groupby([over, col], as_index=False).mean()
    return df.pivot(over, col, val).corr()

def print_correlations(activity):
    "Skriv ut olika relevanta korrelationsvärden"
    print('--------------------------')
    print("Correlation for yearweek:")
    print(correlations(activity, 'yearweek', 'Type', 'movement'))
    print("----")
    print("Correlation for weekday:")
    print(correlations(activity, 'weekday', 'Type', 'movement'))
    print("----")
    print("Correlation for hour:")
    print(correlations(activity, 'hour', 'Type', 'movement'))
    print('--------------------------')
    return activity

# Skapar lättviktiga objekt för att innehålla datat skapat ovan samt för korsvalidering av modeller 
Prediction = namedtuple('Prediction', ['activity', 'results'])
ActivityData = namedtuple('ActivityData', ['features', 'target'])
ModelAccuracy = namedtuple('ModelAccuracy', ['name', 'mu', 'sigma'])

def extract_features(activity):
    "Normalisera och behåll endast de prediktorer vi vill träna modellen på"

    # Ta bort sensorvärdena från loungen som har låg korrelation 
    # med de andra typerna och inte bidrar med mycket data ändå
    activity = activity.loc[activity['Type'] != 'lounge']
    # Ta bort kolumner från tidigare stadier helt (behåll bara time & movement)
    activity = activity.loc[:, ['time', 'movement']]
    # Sätt tidsbaserat index för downsampling (konvertera stickprovsupplösningen)
    activity = activity.set_index('time')
    # Summera alla movement per timma
    activity = activity.resample('1H').sum()

    # Hämta olika värden för varje post
    activity['yearweek'] = activity.apply(lambda row: row.name.weekofyear, axis=1)
    activity['weekday'] = activity.apply(lambda row: row.name.dayofweek, axis=1)
    activity['hour'] = activity.apply(lambda row: row.name.hour, axis=1)

    # Ta bort movement-kolumnen då vi har resamplat per timme
    target = activity['movement'].values
    features = activity.drop('movement', axis=1).values

    # Konverta alla värden till en siffra mellan 0 och 1
    features = preprocessing.MaxAbsScaler().fit_transform(features)

    return Prediction(ActivityData(features, target), [])

def train_knn(prediction):
    n = 7
    knn = neighbors.KNeighborsClassifier(n)
    cv = model_selection.cross_val_score(
        knn,
        prediction.activity.features,
        prediction.activity.target,
        cv=5
    )
    prediction.results.append(ModelAccuracy(
        'kNN({})'.format(n),
        cv.mean(),
        cv.std()
    ))
    return prediction

def print_results(prediction):
    for result in prediction.results:
        print('{} accuracy: {:.2f} (± {:.2f})'.format(
            result.name,
            result.mu,
            result.sigma
        ))

# Enbart felsökning
def set_global_activity_var(df):
    global activity
    activity = df
    return df

# Alla funktioner som ska köras finns i PARTS-listan
PARTS = [get_csv_data, prepare_data, print_some, set_global_activity_var]
PARTS.append(plot_data)
PARTS.append(print_correlations)
PARTS.append(extract_features)
PARTS.append(train_knn)
PARTS.append(print_results)

# Kör funktionerna i PARTS
functools.reduce(lambda prev, part: part(prev), PARTS, 'yanzi_motion_20000l_slump.csv')
