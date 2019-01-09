from collections import namedtuple
import functools
import logging
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
import matplotlib.pyplot as plt

LOG = logging.getLogger('yanzi_plot')

def configure_logging():
    global LOG
    LOG = logging.getLogger('yanzi_plot')
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(levelname)-8s %(asctime)s: %(message)s'
    ))
    LOG.handlers = [handler]
    LOG.setLevel(logging.DEBUG)

def get_csv_data(filename):
    "Läs in CSV-fil"
    LOG.info('Läs CSV-fil %s', filename)
    return pd.read_csv(filename, sep=';', encoding = "utf-8")

def add_rooms(df):
    "Kombinera typ av plats för sensorn med aktivitet"
    sensorlist = get_csv_data('sensor_list.csv')
    return df.join(sensorlist.set_index('Sensor'), on='sensorId')

def prepare_data(activity):
    "Parsea tidsformat, skapa/plocka ut intressanta kolumner"
    LOG.info('Förbered data')

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
    
    # För att hålla koll på vilken kvart det är
    activity['weekdayhourquarter'] = activity.apply(lambda row: row['weekday']*12*4 + row['hour']*4 + row['minute']//4, axis=1)

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
    LOG.info('Korrelera över %s', over)
    df = df.groupby([over, col], as_index=False).mean()
    return df.pivot(over, col, val).corr()

def print_correlations(activity):
    "Skriv ut olika relevanta korrelationsvärden"
    print('--------------------------')
    print("Korrelation för vecka på året:")
    print(correlations(activity, 'yearweek', 'Type', 'movement'))
    print("----")
    print("Korrelation för veckodag:")
    print(correlations(activity, 'weekday', 'Type', 'movement'))
    print("----")
    print("Korrelation för timme:")
    print(correlations(activity, 'hour', 'Type', 'movement'))
    print('--------------------------')
    
    # Kontrollera korrelationen mellan sensorer
    sensor_correlations_print(activity)
    return activity

def sensor_correlations_print(activity):
    print("----")
    print("Korrelation för timme (eller kvart, förresten?):")
    sensor_correlations = correlations(activity, 'weekdayhourquarter', 'sensorId', 'movement')
    
    print('--------------------------')
    # Plotta korrelationsmatris
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(sensor_correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,81,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(sensor_correlations.columns.values)
    ax.set_yticklabels(sensor_correlations.columns.values)
    plt.show()

    return activity

# Skapa lättviktigt objekt som håller prediktorer och movement separat
ActivityData = namedtuple('ActivityData', ['features', 'target'])

class Prediction:
    def __init__(self, activity):
        self.activity = activity

    def set_models(self, a, b):
        self.A = a
        self.B = b
        self.both = [a, b]
        return self

sensor_encoder = preprocessing.OneHotEncoder(sparse=False)
feature_scaler = preprocessing.MinMaxScaler()

def extract_features(activity):
    "Normalisera och behåll endast de prediktorer vi vill träna modellen på"

    LOG.info('Plocka ut prediktorer')

    # Skapa en kolumn för varje sensor (1/0 beroende på vilken det är)
    LOG.debug('One-hot coding sensorId')
    # sensors = numpy-array
    sensors = sensor_encoder.fit_transform(
        activity['sensorId'].values.reshape(-1, 1)
    )

    activity = activity.loc[:, ['time', 'sensorId', 'movement']]

    LOG.debug('Skapar kolumner för valda tidsspann')
    
    activity = pd.concat([activity, pd.DataFrame(sensors)], axis=1)

    activity = activity.set_index('time')
    # Den måste registrera 2 movement på 15 minuter för att det ska räknas
    activity = activity.resample('15T').sum() > 2

    activity['weekday'] = activity.apply(lambda row: row.name.dayofweek, axis=1)
    activity['hour'] = activity.apply(lambda row: row.name.hour, axis=1)
    activity['quarter'] = activity.apply(lambda row: row.name.minute //15, axis=1)

    # Ta bort nätter och helger igen då resamplingen lagt till dem, igen – funkar detta, på riktigt?
    activity = activity[activity['hour'] >= 7]
    activity = activity[activity['hour'] < 20]
    activity = activity[activity['weekday'] <= 4]

    LOG.debug('Plocka ut prediktorer och y-värdet')
    target = activity['movement'].values
    features = activity.drop('movement', axis=1).values

    LOG.debug('Skala om alla prediktorer till ett värde mellan 0 och 1')
    features = feature_scaler.fit_transform(features)

    return Prediction(ActivityData(features, target))

class ModelInformation:
    "Hjälpklass som gör det lätt att spot-checka en modell"

    def __init__(self, name, implementation):
        self.name = name
        self.implementation = implementation

    def cross_validate(self, activity):
        LOG.debug('Korsvalidera %s', self.name)
        splitter = model_selection.TimeSeriesSplit(n_splits=5)
        self.cv = model_selection.cross_val_score(
            self.implementation,
            activity.features,
            activity.target,
            cv=splitter
        )

    def train(self, activity):
        LOG.debug('Träna %s', self.name)
        self.implementation.fit(activity.features, activity.target)

    def __str__(self):
        return '{} träffsäkerhet: {:.2f} (± {:.2f})'.format(
            self.name,
            self.cv.mean(),
            self.cv.std()*2
        )

def evaluate_models(prediction):
    LOG.info('Korsvalidera båda modellerna')
    for model in prediction.both:
        model.cross_validate(prediction.activity)
    return prediction

def print_results(prediction):
    for result in prediction.both:
        print(str(result))
    return prediction

def plot_random_day_predictions(prediction):
    "Plotta datat på olika sätt"
    output_file('random.html')

    LOG.info('Träna båda modellerna')
    # Träna modellerna på hela datamängden för att kunna visualisera prognoser
    for model in prediction.both:
        model.train(prediction.activity)

    # Välj ut 40 slumpmässiga sensorer mellan valda tidpunkter
    sensors = np.identity(81)[np.random.randint(0, 81, 40)]
    min_hour = 7
    max_hour = 19
    hour = np.repeat(np.arange(min_hour, max_hour), 4)
    quarter = np.tile(np.arange(0, 4), max_hour - min_hour)

    # Skriver ut felsökningsinformation
    LOG.debug('min_hour = %d, max_hour = %d', min_hour, max_hour)
    figures = list()
    random_week = figure(
        title='Prognos aktivitet en vecka för några sensorer',
        x_axis_label='Veckodag',
        y_axis_label='Timme på dagen',
        x_range=(-1,5),
        y_range=(min_hour-1,max_hour+1)
    )
    figures.append(random_week)
    for colour, model in zip([-1, 1], prediction.both):
        LOG.info('Rita graf för %s', model.name)
        for weekday in range(0,5):
            LOG.debug('..%s dag %d', model.name, weekday)
            for sensor in sensors:
                # Observation för varje kvart på dagen (först välja ut klockslagen)
                times = np.column_stack((
                    np.full(int((max_hour - min_hour)*4), weekday),
                    hour,
                    quarter
                ))
                times = feature_scaler.transform(
                    np.hstack((
                        np.tile(sensor, ((int(max_hour-min_hour)*4), 1)),
                        times
                    ))
                )

                predicted = model.implementation.predict(times)
                pad_top_bottom = lambda arr, bottom, top: \
                    np.concatenate((np.full(1, bottom), arr, np.full(1, top)))

                random_week.patch(
                    # Rita ut det predikterade värdet (mellan två nollor), 
                    # sen rita till vä eller hö beroende på vilken modell det är (colour tar -1 / 1)
                    x=weekday + 0.5 * colour * pad_top_bottom(predicted, 0, 0),
                    y=pad_top_bottom(hour + quarter/4, min_hour-0.25, max_hour + 0.25),
                    fill_color=palettes.Category10_5[colour],
                    color=palettes.Category10_5[colour],
                    alpha=0.05,
                    legend=model.name
                )

    show(column(*figures))

# Enbart felsökning
def set_global_activity_var(df):
    global activity
    activity = df
    return df

def set_global_prediction_var(pred):
    global prediction
    prediction = pred
    return pred

configure_logging()

# Alla funktioner som ska köras finns i PARTS-listan
PARTS = [get_csv_data, prepare_data, set_global_activity_var]
#PARTS.append(plot_data)
PARTS.append(print_correlations)
PARTS.append(extract_features)
PARTS.append(lambda prediction: prediction.set_models(
    ModelInformation('kNN(3)', neighbors.KNeighborsClassifier(3)),
    ModelInformation('kNN(15)', neighbors.KNeighborsClassifier(15))
))
PARTS.append(evaluate_models)
PARTS.append(print_results)
PARTS.append(set_global_prediction_var)
PARTS.append(plot_random_day_predictions)

# Kör funktionerna i PARTS
functools.reduce(
    lambda prev, part: part(prev),
    PARTS,
    'yanzi_motion_nonightweekend_from150000l_slump.csv'
)
