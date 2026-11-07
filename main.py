"""
Package Requirement :
    matplotlib
    numpy == 1.26.4 (for scikit-learn-extra)
    pandas == 2.3.3 (for numpy == 1.26.4)
    tslearn
    pypots
    scikit-learn
    scikit-learn-extra
    # scikit-fda
    # aeon
"""
import datetime

import numpy as np
import pandas as pd
# import skfda
# from skfda.preprocessing.dim_reduction import FPCA  # multimethod < 2.1
# from skfda.representation.basis import BSplineBasis
# from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from database import DataBase
from figure import Figure
from nsb import NightSkyBrightness, KMedoid, Agglomerative, TSKMeans, MiniBatchKMean, PypotsCRLI


DATABASE_PATH = {  # path of dataset relative to main.
    # raw database from online MySQL
    "HKU_RAW": "iyl/sqmle_nsb_data_table/HKU.csv",
    "TST_RAW": "iyl/sqmle_nsb_data_table/TST.csv",
    "AP_RAW": "iyl/sqmle_nsb_data_table/AP.csv",
    
    # database for Sun/Moon rise/set time
    "Sun_rise_set": "src/Sun/Sun_rise_set_2018-2026.csv",
    "Moon_rise_set": "src/Moon/Moon_rise_set_2018-2026.csv",
    
    # dataset filtered for Sun set/rise
    "HKU_sun_rise_set": "src/HKU/HKU_RAW.csv",
    "TST_sun_rise_set": "src/TST/TST_RAW.csv",
    
    # dataset filtered for Sun set/rise
    # with Moon phase and illuminated ratio injected
    "HKU_sun_moon": "src/HKU/HKU.csv",
    "TST_sun_moon": "src/TST/TST.csv",
}
MONTH = {
    "01": "Jan",
    "02": "Feb",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "Aug",
    "09": "Sep",
    "10": "Oct",
    "11": "Nov",
    "12": "Dec"
}


def custom_collect_data_sun_set_rise(
        nsb_db_path,
        sun_rise_set_db_path,
        moon_rise_set_db_path,
        export_file_name  # relative path with file name, no extension
):
    """
    Extract data from 2018 to 2026
    filter to get data only from sun set to sun rise
    column "received_adjusted" is split into 2 columns "date" and "time"
    only write column "date", "time", "nsb" to new file
    
    :param nsb_db_path:
    :param sun_rise_set_db_path:
    :param moon_rise_set_db_path:
    :param export_file_name:
    :return:
    """
    import pandas as pd
    
    NSB = NightSkyBrightness(
            nsb_db_path,
            sun_rise_set_db_path,
            moon_rise_set_db_path
    )
    db_list = []  # yearly dataframe
    
    for year in range(2018, 2026 + 1):  # range of year to collect
        _db_list = []  # 1-year data

        for month in range(1, 12 + 1):
            # get number of days in a month
            month1 = datetime.date(year = year, month = month, day = 1)
            month2 = datetime.date(year = year if month + 1 <= 12 else year + 1, month = (month + 1) % 12 if month + 1 > 12 else month + 1, day = 1)
            days = month2 - month1
            days = days.days

            for day in range(1, days + 1):
                # data from sun set to sun rise
                df = NSB.get_one_night_data(f"{year}-{month:02}-{day:02}")[2]  # pandas.DataFrame

                # convert str datetime to Python datetime
                df["received_adjusted"] = pd.to_datetime(df["received_adjusted"])

                # separate datetime into 2 columns: date, time
                df["date"] = df["received_adjusted"].dt.date
                df["time"] = df["received_adjusted"].dt.time

                # append to data structure
                _db_list.append(df[["date", "time", "nsb"]])  # internal

                del df

        # export the yearly dataframe (just in case)
        _db = DataBase().concat(_db_list)
        DataBase().export(f"{export_file_name}_{year}.csv", _db)
        # append to data structure
        db_list.append(_db)

    # merge all gathered data into 1 dataframe
    db = DataBase().concat(db_list)  # pandas.DataFrame
    del db_list
    
    # export this dataframe for convenience future processing
    DataBase().export(f"{export_file_name}.csv", dataframe = db)
    
    return db  # pandas.DataFrame


def custom_db_insert_moon_data(
        nsb_db_path,
        sun_rise_set_db_path,
        moon_rise_set_db_path,
        export_file_name  # relative path with file name and extension
):
    """
    After custom_collect_data_sun_set_rise()
    insert astronomical_date --> group each night's date into single calendar date
    insert moon phase and illuminated ratio
    
    :param nsb_db_path:
    :param sun_rise_set_db_path:
    :param moon_rise_set_db_path:
    :param export_file_name:
    :return:
    """
    import pandas as pd
    
    NSB = NightSkyBrightness(
            nsb_db_path,
            sun_rise_set_db_path,
            moon_rise_set_db_path
    )
    db = NSB.nsb_db.read()  # DataBase(nsb_db_path).read()
    
    # convert date after midnight to previous date --> match the same astronomical night
    _db_column_time = pd.to_datetime(db["date"] + " " + db["time"])  # convert whole column to datetime
    _filters = (
            (_db_column_time.dt.time >= pd.to_datetime("00:00").time()) &
            (_db_column_time.dt.time <= pd.to_datetime("08:00").time())
    )
    db["astronomical_date"] = db["date"]  # copy from calendar date
    # convert date after midnight to previous date in column "astronomical_date"
    db.loc[_filters, "astronomical_date"] = (_db_column_time[_filters] - pd.Timedelta(days = 1)).dt.strftime("%Y-%m-%d")
    
    # tag moon phase to each night
    db["moon_phase"] = db["astronomical_date"].apply(lambda x: NSB.get_moon_phase(x)[0])
    db["moon_illuminated_ratio"] = db["astronomical_date"].apply(lambda x: NSB.get_moon_phase(x)[1])
    
    # export result
    DataBase().export(export_file_name, dataframe = db)
    
    return db


def custom_preprocess_data(nsb_db_path, interpolate = True):
    """
    After custom_db_insert_moon_data()
    Filter data from fixed sun set time to sun rise time
    Align data to a fixed timeline (time interval = 1s), missing data are filled with NaN
    Drop data with number of data points less than data_length_tolerance per night
    Drop night with unrealistic value of night sky brightness, outside range of (10, 23) mag arcsec-2
    (optional) Interpolate NaN data without stretching / compressing
    Store data and features into an array
    
    :param nsb_db_path: str
    :param interpolate: bool
    :return: tuple (dataset, date, time, moon_illuminated_ratio, moon_phase)
    """
    dataset = []
    date = []
    time = []
    moon_illuminated_ratio = []
    moon_phase = []
    data_drop_count = 0
    data_length_tolerance = 3 * 60*2  # minimum number of data points per night,  # 30s 1 data point, at least 3 hours of data
    sun_set_time = "19:00:00"
    sun_rise_time = "05:30:00"
    
    db_class = DataBase(nsb_db_path)
    db = db_class.read().drop_duplicates()
    
    # Pivot Data: Reshape into a nightly curve matrix
    # [       time1 time2 ...
    #   date1 nsb11 nsb12 ...
    #   date2 nsb21 nsb22 ... ]
    db_matrix = db.pivot(
            index = "astronomical_date",
            columns = "time",
            values = "nsb"
    )
    # reorder matrix column "time" to match physical chronological order
    _matrix_column = db_matrix.columns
    _col_first = [i for i in _matrix_column if i >= "16:00:00"]
    _col_first.sort()
    _col_last = [i for i in _matrix_column if i <= "08:00:00"]
    _col_last.sort()
    
    if len(_col_first) + len(_col_last) == len(_matrix_column):
        db_matrix = db_matrix[_col_first + _col_last]
    
    else:
        raise ValueError("[ERROR] in reordering matrix columns")
    
    # drop column according to fixed sun set/rise time
    _filter = (
        (db_matrix.columns >= sun_set_time) |
        (db_matrix.columns <= sun_rise_time)
    )
    db_matrix = db_matrix.loc[:, _filter]
    
    # save the time
    time = db_matrix.columns
    
    print("[INFO] Collecting data")
    
    for _date, _group in db.groupby("astronomical_date"):
        _data = np.copy( db_matrix.loc[_date].values )  # numpy array
        
        if len(_data[~np.isnan(_data)]) <= data_length_tolerance:
            # drop night with insufficient data points
            data_drop_count += 1
            continue
        
        if np.nanmin(_data) <= 10.0 or np.nanmax(_data) >= 23.0:
            # drop night with unrealistic value of night sky brightness
            # outside range of (10, 23) mag arcsec-2
            data_drop_count += 1
            continue
        
        # linear interpolate data
        # https://stackoverflow.com/questions/38064697/interpolating-a-numpy-array-to-fit-another-array
        # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
        if interpolate:
            _data[np.isnan(_data)] = np.interp(
                    np.isnan(_data).nonzero()[0],
                    (~np.isnan(_data)).nonzero()[0],
                    _data[~np.isnan(_data)]
            )
        
        # append to data structure
        dataset.append(_data)
        date.append(_date)
        moon_illuminated_ratio.append(_group["moon_illuminated_ratio"][_group["moon_illuminated_ratio"].keys()[0]])
        moon_phase.append(_group["moon_phase"][_group["moon_phase"].keys()[0]])
    
    del db_class
    del db
    del db_matrix
    dataset = np.array(dataset, dtype = np.float32)
    date = np.array(date)
    moon_illuminated_ratio = np.array(moon_illuminated_ratio, dtype = np.float32)
    moon_phase = np.array(moon_phase)
    print(f"[INFO] Dropped {data_drop_count} night(s) of data")
    print(f"[INFO] Size of dataset : {dataset.shape}")
    
    return dataset, date, time, moon_illuminated_ratio, moon_phase


# custom_collect_data_sun_set_rise(
#         DATABASE_PATH["TST_RAW"],
#         DATABASE_PATH["Sun_rise_set"],
#         DATABASE_PATH["Moon_rise_set"],
#         "src/TST/TST"
# )
# custom_db_insert_moon_data(
#         DATABASE_PATH["TST_sun_rise_set"],
#         DATABASE_PATH["Sun_rise_set"],
#         DATABASE_PATH["Moon_rise_set"],
#         "src/TST/TST.csv"
# )

dataset, date, time, moon_illuminated_ratio, moon_phase = custom_preprocess_data(
        DATABASE_PATH["HKU_sun_moon"],
        interpolate = True
)


# CRLI @ Pypots (run out of computer memory)
# import torch
# print(torch.cuda.is_available())
# Initialized PyPOTS Ecosystem configuration file C:\\Users\\PAUL\\.pypots\\config.ini successfully.

# down sampling
# take mean for each segment
print("[INFO] Down sampling")
chunks = np.array_split(dataset, 1000, axis = 1)  # each night's data is split into 1200 segments
compressed_dataset = []

for chunk in chunks:
    compressed_dataset.append(
            np.nanmean(chunk, axis = 1, keepdims = True)
    )

compressed_dataset = np.squeeze( np.concatenate(compressed_dataset, axis = 1) )
# compressed_dataset = np.nan_to_num(compressed_dataset, nan = 0.0)  # fill NaN with 0.0

# reshape into PyPOTS 3D array format: (n_samples, n_steps, n_features)
# Convert to the final 3D NumPy array for CRLI
X = np.expand_dims(dataset, axis = -1)  # compressed_dataset[:, :, np.newaxis]  # 3D array

train_indices = np.random.choice(X.shape[0], size = 2000, replace = False)
X_train = X[train_indices]
train_dataset = {
    "X": X_train
}
all_dataset = {
    "X": X
}

# STEP 4
print(f"[INFO] Shape of dataset : {dataset.shape}")
n_samples = dataset.shape[0]  # number of nights
n_steps = dataset.shape[1]    # number of data points per night
n_features = 1  # night-sky brightness

print("[INFO] Train CRLI model")
crli = PypotsCRLI()
crli.n_samples = n_samples
crli.n_steps = n_steps
crli.n_clusters = 12
crli.batch_size = 128  # todo memory issue
crli.rnn_hidden_size = 32
crli.epoch = 10

# optimal_k = crli.find_optimal_k((10, 30), train_set = train_dataset, validation_set = train_dataset)

# train actual CRLI model
# crli.auto_partial_fit(dataset = dataset, save_model = True)
crli.fit(train_set = train_dataset, save_model = True)

# result = crli.MODEL.predict("temp_dataset.h5")
result = crli.MODEL.predict(all_dataset, batch_size = crli.batch_size)
label = result["clustering"]  # cluster ID corresponding to db_HKU_pivot.index

# STEP 5
print("[INFO] Step 5 : Plot result")
print(label)

# plot result
print("[INFO] Plotting result")
fig = Figure("CRLI Clustering\nHKU, 2018-2026, 19:00-05:30\n$I_{avg} = average moon illuminated ratio", 4, 6)

for cluster_id in range(crli.n_clusters):
    cluster_data = dataset[label == cluster_id].tolist()
    _moon_illuminated_ratio_data = moon_illuminated_ratio[label == cluster_id].tolist()
    
    index = np.random.choice(len(cluster_data), size = min(50, len(cluster_data)), replace = False)
    print(f"Plotting cluster {cluster_id}")
    
    for i in index:
        fig.plot(
                np.linspace(0, 1, len(cluster_data[i])),
                cluster_data[i],
                axes_index = cluster_id,
                color = "grey",
                alpha = 0.1
        )
    
    fig.set_axes_properties(
            axes_index = cluster_id,
            title = f"Cluster {cluster_id}\n" + "$I_{avg}$ = " + f"{sum(_moon_illuminated_ratio_data) / len(_moon_illuminated_ratio_data):.1%}",
            xlabel = "time",
            ylabel = "NSB",
            ylim = [13, 19],
            # xticks = [0, len(time) - 1],
            # xticklabels = [],
            invert_yaxis = True,
            # legend = True
    )

fig.show()

"""
# Mini-batch K-Means
k_medoids = MiniBatchKMean()  # class object

print("[INFO] Splitting dataset for training")
# dataset_scaled = TimeSeriesScalerMeanVariance().fit_transform(dataset)[:, :, 0].copy()

k_medoids.dataset_with_metadata(
        dataset,
        {
            "date"                  : date,
            "moon_illuminated_ratio": moon_illuminated_ratio,
            "moon_phase"            : moon_phase
        },
        0.7,
        0.15
)


# find optimal k
optimal_k, train_silhouette_score = k_medoids.find_optimal_k((10, 30), batch_size = 256)  # todo
print(f"Train score : {train_silhouette_score}")

# train actual model
k_medoids.fit(optimal_k, batch_size = 256, save_model = False)

# test on unseen data
test_label, test_silhouette_score = k_medoids.predict()

# apply model to entire dataset
cluster_labels = k_medoids.MODEL.fit_predict(dataset)
# cluster_labels, dataset_silhouette_score = k_medoids.predict(test_set = dataset)


# plot result
print("[INFO] Plotting result")
fig = Figure("K-Medoids Clustering\nHKU, 2018-2026, 19:00-05:30", 3, 5)

for cluster_id, center_data in enumerate(k_medoids.MODEL.cluster_centers_):
    cluster_data = dataset[cluster_labels == cluster_id].tolist()
    index = np.random.choice(len(cluster_data), size = min(50, len(cluster_data)), replace = False)
    print(f"Plotting cluster {cluster_id}")
    
    for i in index:
        fig.plot(
                np.linspace(0, 1, len(cluster_data[i])),
                cluster_data[i],
                axes_index = cluster_id,
                color = "grey",
                alpha = 0.1
        )
    
    fig.plot(
            np.linspace(0, 1, len(center_data)),
            center_data,
            axes_index = cluster_id,
            label = f"Center {cluster_id}",
            color = "red"
    )
    
    fig.set_axes_properties(
            axes_index = cluster_id,
            title = f"Cluster {cluster_id}",
            xlabel = "time",
            ylabel = "NSB",
            ylim = [13, 19],
            # xticks = [0, len(time) - 1],
            # xticklabels = [],
            invert_yaxis = True,
            # legend = True
    )

fig.show()
"""
"""
# Time Series K-Means
ts_kmeans = TSKMeans()  # class object

print("[INFO] Spllitting dataset for training")
dataset = TimeSeriesScalerMeanVariance(mu = 0, std = 1).fit_transform(dataset).squeeze()

ts_kmeans.dataset_with_metadata(
        dataset,
        {
            "date": date,
            "moon_illuminated_ratio": moon_illuminated_ratio,
            "moon_phase": moon_phase
        },
        0.7,
        0.15
)

# find optimal k
metric = "euclidean"
optimal_k, train_silhouette_score = ts_kmeans.find_optimal_k((10, 30), metric = metric)  # todo
print(f"Train score : {train_silhouette_score}")

# train actual model
ts_kmeans.fit(optimal_k, metric = metric, save_model = False)

# test on unseen data
test_label, test_silhouette_score = ts_kmeans.predict(metric = metric)

# apply model to entire dataset
cluster_labels = ts_kmeans.MODEL.fit_predict(dataset)
# cluster_labels, dataset_silhouette_score = ts_kmeans.predict(test_set = dataset)


# plot result
print("[INFO] Plotting result")
fig = Figure("TimeSeries K-Means Clustering\nHKU, 2018-2026, 19:00-05:30", 3, 5)

for cluster_id, cluster_center in enumerate(ts_kmeans.MODEL.cluster_centers_):
    center_data = cluster_center
    cluster_data = dataset[cluster_labels == cluster_id].tolist()
    index = np.random.choice(len(cluster_data), size = min(50, len(cluster_data)), replace = False)
    print(f"Plotting cluster {cluster_id}")
    
    for i in index:
        fig.plot(
                np.linspace(0, 1, len(cluster_data[i])),
                cluster_data[i],
                axes_index = cluster_id,
                color = "grey",
                alpha = 0.1
        )
    
    fig.plot(
            np.linspace(0, 1, len(center_data)),
            center_data,
            axes_index = cluster_id,
            label = f"Center {cluster_id}",
            color = "red"
    )
    
    fig.set_axes_properties(
            axes_index = cluster_id,
            title = f"Cluster {cluster_id}",
            xlabel = "time",
            ylabel = "NSB",
            ylim = [13, 19],
            # xticks = [0, len(time) - 1],
            # xticklabels = [],
            invert_yaxis = True,
            # legend = True
    )

fig.show()
"""
"""
# Agglomerative Clustering
agglomerative = Agglomerative()
# dataset_scaled = TimeSeriesScalerMeanVariance().fit_transform(dataset)[:, :, 0]
agglomerative.dataset_with_metadata(
        dataset,
        {
            "date": date,
            "moon_illuminated_ratio": moon_illuminated_ratio,
            "moon_phase": moon_phase
        },
        0.7,
        0.15
)
# model, n_clusters, cluster_labels = agglomerative.fit_predict(
#         distance_threshold = 1000.0,
#         save_model = False
# )
model, n_clusters, cluster_labels = agglomerative.fit_predict(
        n_clusters = 28,
        save_model = False,
)
print(f"[INFO] Final number of clusters: {n_clusters}")


# plot result
from collections import Counter
print("[INFO] Plotting result")
fig = Figure("Agglomerative Clustering\nHKU, 2018-2026, 19:00-05:30\n$I_{avg}$ = average moon illuminated ratio",
             4, 7
)

for cluster_id in range(n_clusters):
    cluster_data = dataset[cluster_labels == cluster_id].tolist()
    _moon_illuminated_ratio_data = moon_illuminated_ratio[cluster_labels == cluster_id].tolist()
    _month = date[cluster_labels == cluster_id].tolist()
    _month = [i.split("-")[1] for i in _month]
    _common_month = Counter(_month).most_common(2)
    
    index = np.random.choice(len(cluster_data), size = min(50, len(cluster_data)), replace = False)
    print(f"Plotting cluster {cluster_id}")
    
    for i in index:
        fig.plot(
                np.linspace(0, 1, len(cluster_data[i])),
                cluster_data[i],
                axes_index = cluster_id,
                color = "grey",
                alpha = 0.1
        )
    
    fig.set_axes_properties(
            axes_index = cluster_id,
            title = f"Cluster {cluster_id} " +
                    "($I_{avg}$ = " + f"{sum(_moon_illuminated_ratio_data) / len(_moon_illuminated_ratio_data):.1%})\n" +
                    f"{MONTH[ _common_month[0][0] ]}: {_common_month[0][1] / len(_month):.1%}; {MONTH[ _common_month[1 if len(_common_month) > 1 else 0][0] ]}: {_common_month[1 if len(_common_month) > 1 else 0][1] / len(_month):.1%}",
            xlabel = "time",
            ylabel = "NSB",
            ylim = [13, 19],
            # xticks = [0, len(time) - 1],
            # xticklabels = [],
            invert_yaxis = True,
            # legend = True
    )

fig.show()
"""
'''
# K-Medoids clustering (updated)
k_medoids = KMedoid()  # class object

print("[INFO] Spllitting dataset for training")
k_medoids.dataset_with_metadata(
        dataset,
        {
            "date": date,
            "moon_illuminated_ratio": moon_illuminated_ratio,
            "moon_phase": moon_phase
        },
        0.7,
        0.15
)

"""
# apply FPCA
print("[INFO] Applying FPCA")
domain_range = (0, len(time))

fd_train = skfda.FDataGrid(data_matrix = k_medoids.TRAIN_SET, grid_points = np.arange(len(time)), domain_range = domain_range)
fd_validation = skfda.FDataGrid(data_matrix = k_medoids.VALIDATION_SET, grid_points = np.arange(len(time)), domain_range = domain_range)
fd_test = skfda.FDataGrid(data_matrix = k_medoids.TEST_SET, grid_points = np.arange(len(time)), domain_range = domain_range)

# use n_basis number of smooth curve to represent each night's data
n_basis = 17  # 17, 7
basis = BSplineBasis(domain_range = domain_range, n_basis = n_basis)

fd_train_basis = fd_train.to_basis(basis)
fd_validation_basis = fd_validation.to_basis(basis)
fd_test_basis = fd_test.to_basis(basis)

# take n_components number of major components (eigenfunction)
n_components = 3  # 3, 4
fpca = FPCA(n_components = n_components)
fpca.fit(fd_train_basis)  # numpy.core._exceptions._ArrayMemoryError

fpca_train = fpca.transform(fd_train_basis)
fpca_validation = fpca.transform(fd_validation_basis)
fpca_test = fpca.transform(fd_test_basis)

k_medoids.TRAIN_SET = fpca_train
k_medoids.VALIDATION_SET = fpca_validation
k_medoids.TEST_SET = fpca_test
"""
"""
# Apply Incremental PCA
print("[INFO] Applying Incremental PCA")
ipca = IncrementalPCA(n_components = 4)  # HKU:4, TST:2/7

batch_size = 100
for i in range(0, k_medoids.TRAIN_SET.shape[0], batch_size):
    batch_x = k_medoids.TRAIN_SET[i:i+batch_size, :]
    ipca.partial_fit(batch_x)

ipca_train = ipca.transform(k_medoids.TRAIN_SET)
ipca_validation = ipca.transform(k_medoids.VALIDATION_SET)
ipca_test = ipca.transform(k_medoids.TEST_SET)

k_medoids.TRAIN_SET = ipca_train
k_medoids.VALIDATION_SET = ipca_validation
k_medoids.TEST_SET = ipca_test
"""
"""
# Apply PCA
print("[INFO] Applying PCA")
scaler = StandardScaler()
scaled_train = scaler.fit_transform(k_medoids.TRAIN_SET)
scaled_validation = scaler.fit_transform(k_medoids.VALIDATION_SET)
scaled_test = scaler.fit_transform(k_medoids.TEST_SET)

n_components = 5  # extract number of component to see variance distribution
pca = PCA(n_components = n_components, random_state = 42)

pca_train = pca.fit_transform(scaled_train)
pca_validation = pca.fit_transform(scaled_validation)
pca_test = pca.fit_transform(scaled_test)

k_medoids.TRAIN_SET = pca_train
k_medoids.VALIDATION_SET = pca_validation
k_medoids.TEST_SET = pca_test
"""

# find optimal k
metric = "euclidean"
optimal_k, train_silhouette_score = k_medoids.find_optimal_k((10, 30), metric = metric)  # todo
print(f"Train score : {train_silhouette_score}")

# train actual model
k_medoids.fit(optimal_k, metric = metric, save_model = False)

# test on unseen data
test_label, test_silhouette_score = k_medoids.predict(metric = metric)

# apply model to entire dataset
cluster_labels = k_medoids.MODEL.fit_predict(dataset, metric = metric)
# cluster_labels, dataset_silhouette_score = k_medoids.predict(test_set = dataset)


# plot result
print("[INFO] Plotting result")
fig = Figure("K-Medoids Clustering\nHKU, 2018-2026, 19:00-05:30", 3, 5)

for cluster_id, medoid_index in enumerate(k_medoids.MODEL.medoid_indices_):
    medoid_date = date[medoid_index]
    medoid_data = dataset[medoid_index]
    cluster_data = dataset[cluster_labels == cluster_id].tolist()
    index = np.random.choice(len(cluster_data), size = min(50, len(cluster_data)), replace = False)
    print(f"Plotting cluster {cluster_id}")
    
    for i in index:
        fig.plot(
                np.linspace(0, 1, len(cluster_data[i])),
                cluster_data[i],
                axes_index = cluster_id,
                color = "grey",
                alpha = 0.1
        )
    
    fig.plot(
            np.linspace(0, 1, len(medoid_data)),
            medoid_data,
            axes_index = cluster_id,
            label = f"Medoid {cluster_id}",
            color = "red"
    )
    
    fig.set_axes_properties(
            axes_index = cluster_id,
            title = f"Cluster {cluster_id}",
            xlabel = "time",
            ylabel = "NSB",
            ylim = [13, 19],
            # xticks = [0, len(time) - 1],
            # xticklabels = [],
            invert_yaxis = True,
            # legend = True
    )

fig.show()
'''


"""
HKU = NightSkyBrightness(
        "iyl/sqmle_nsb_data_table/HKU.csv",
        "src/Sun/Sun_rise_set_2018-2026.csv",
        "src/Moon/Moon_rise_set_2018-2026.csv"
)
AP = NightSkyBrightness(
        "iyl/sqmle_nsb_data_table/AP.csv",
        "src/Sun/Sun_rise_set_2018-2026.csv",
        "src/Moon/Moon_rise_set_2018-2026.csv"
)

date = "2025-06-01"
HKU_data = HKU.get_one_night_data(date)
AP_data = AP.get_one_night_data(date)

fig = Figure("Single-night NSB")
fig.plot(
        HKU_data[0],
        HKU_data[1],
        label = "HKU"
)
fig.plot(
        HKU_data[0],
        AP_data[1],
        label = "AP"
)

tick_index = [0] + [int(i/7 * len(HKU_data[0])) - 1 for i in range(1, 8)]
tick = [HKU_data[0].to_list()[i] for i in tick_index]
tick_label = [f"{i[11:13]}:{i[14:16]}" for i in tick]

fig.set_axes_properties(
        title = "2025-06-01",
        xticks = tick,
        xticklabels = tick_label,
        invert_yaxis = True,
        legend = True
)
fig.show()
"""

"""
import pandas as pd

NSB = NightSkyBrightness(
        "src/HKU_RAW.csv",
        "src/Sun/Sun_rise_set_2018-2026.csv",
        "src/Moon/Moon_rise_set_2018-2026.csv"
)

db_HKU = DataBase("src/HKU_RAW.csv").read()

# convert date after midnight to previous date --> match the same astronomical night
_db_column_time = pd.to_datetime(db_HKU["date"] + " " + db_HKU["time"])  # convert whole column to datetime
_filters = (
        (_db_column_time.dt.time >= pd.to_datetime("00:00").time()) &
        (_db_column_time.dt.time <= pd.to_datetime("08:00").time())
)
db_HKU["astronomical_date"] = db_HKU["date"]  # copy from calendar date
# convert date after midnight to previous date in column "astronomical_date"
db_HKU.loc[_filters, "astronomical_date"] = (_db_column_time[_filters] - pd.Timedelta(days = 1)).dt.strftime("%Y-%m-%d")
# print(db_HKU.loc[db_HKU["astronomical_date"] == "2018-01-01", "nsb"])

# tag moon phase to each night
db_HKU["moon_phase"] = db_HKU["astronomical_date"].apply(lambda x: NSB.get_moon_phase(x)[0])
db_HKU["moon_illuminated_ratio"] = db_HKU["astronomical_date"].apply(lambda x: NSB.get_moon_phase(x)[1])

# export result
DataBase().export("src/HKU.csv", dataframe = db_HKU)
"""

"""
def plot_one_month_nsb(fig, month_average, daily_average, moon_phase, title, axes_index = 0):
    fig.plot([i for i in range(1, len(daily_average) + 1)], daily_average, axes_index, label = "NSB")
    fig.plot_horizontal_line(
            month_average,
            axes_index = axes_index,
            label = "Month average NSB",
            linestyle = "--",
            color = "red"
    )
    
    moon_phase = enumerate(moon_phase, start = 1)
    phase = ""
    
    for i in moon_phase:
        
        if phase != i[1][0]:
            fig.plot_vertical_line(
                    i[0],
                    axes_index = axes_index,
                    label_in_graph = i[1][0],
                    linestyle = "--",
                    color = "grey"
            )
            phase = i[1][0]
    
    fig.set_axes_properties(
            axes_index = axes_index,
            title = title,
            xlabel = "Day",
            ylabel = "NSB",
            # xticks = [i * division_length for i in range(division_number)] + [len(x_AP) - 1],
            # xticklabels = ["0", "100"],
            # ylim = [min(daily_average), max(daily_average) + 1],
            # ygrid = True,
            invert_yaxis = True,
            legend = True
    )


# plot 2025 January to December average
NSB_HKU_2025 = NightSkyBrightness(
        "iyl/sqmle_nsb_data_table/HKU.csv",
        "src/Sun/Sun_rise_set_2018-2026.csv",
        "src/Moon/Moon_rise_set_2018-2026.csv"
)

data = {}

for i in range(1, 12 + 1):
    data[i] = NSB_HKU_2025.one_month_average(f"2025-{i:02}-01")


fig = Figure("2025 NSB at HKU", 3, 4)

for i in range(12):  # [0, 11]
    plot_one_month_nsb(
            fig,
            data[i + 1][0],
            data[i + 1][1],
            data[i + 1][2],
            title = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][i],
            axes_index = i
    )
    
fig.show()


# Plot 2025 January and June average
fig = Figure("2025 NSB", 1, 2)
# JANUARY
plot_one_month_nsb(fig, data[1][0], data[1][1], data[1][2], "January", 0)
# JUNE
plot_one_month_nsb(fig, data[6][0], data[6][1], data[6][2], "June", 1)

fig.show()


# Plot 2015 to 2025 January average
fig = Figure("NSB in January from 2015 to 2025", 2, 4)
year = [i for i in range(2018, 2026)]

for index in range(len(year)):
    NSB_HKU = NightSkyBrightness(
            "iyl/sqmle_nsb_data_table/HKU.csv",
            f"src/Sun/Sun_rise_set_{year[index]}.csv",
            f"src/Moon/Moon_rise_set_{year[index]}.csv"
    )
    data = NSB_HKU.one_month_average(f"{year[index]}-01-01")
    plot_one_month_nsb(fig, data[0], data[1], data[2], year[index], index)
    
fig.show()
"""

"""
def prepare_single_night_data(_db, datetime_from, datetime_to):
    # short-term data
    data = _db.get_column(
            ["received_adjusted", "nsb"],
            filters = [
                {"received_adjusted": [">=", datetime_from]},
                {"received_adjusted": ["<=", datetime_to]},
                # {"nsb": ["<", 23]}
            ]
    )
    
    x = data["received_adjusted"]
    y = data["nsb"]
    
    del data
    return x, y


def prepare_daily_avg_data(_db, date_from, date_to, time_from, time_to):
    # 1-month daily avg
    # long-term data
    x = []
    y = []
    
    date_from = datetime.datetime.strptime(date_from, "%Y-%m-%d")
    date_to = datetime.datetime.strptime(date_to, "%Y-%m-%d")
    now = date_from
    
    while now <= date_to:
        date = now.strftime("%Y-%m-%d")  # str
        date1 = now + datetime.timedelta(days = 1)  # datetime.datetime
        date1 = date1.strftime("%Y-%m-%d")  # str
        
        x.append(date)
        
        y_data = _db.get_column(
                ["received_adjusted", "nsb"],
                filters = [
                    {"received_adjusted": [">=", f"{date} {time_from}"]},
                    {"received_adjusted": ["<=", f"{date1} {time_to}"]},  # assuming getting night-time data only
                    # {"nsb": ["<", 23]}
                ]
        )
        
        if not y_data.empty:
            y.append(sum(y_data["nsb"]) / len(y_data["nsb"]))  # average of the daily NSB
        
        else:
            y.append(0)
        
        now += datetime.timedelta(days = 1)
        
    return x, y


def prepare_monthly_avg_data(_db, date_from, date_to, time_from, time_to):
    x = []  # months
    y = []
    
    date_from = datetime.datetime.strptime(date_from, "%Y-%m-%d")
    date_to = datetime.datetime.strptime(date_to, "%Y-%m-%d")
    now = date_from
    
    while now <= date_to:
        date = now.strftime("%Y-%m-%d")  # str
        _month = now.month + 1
        _year = now.year + _month // 13
        _month = _month % 12 or 12
        date1 = datetime.datetime(year = _year, month = _month, day = now.day)  # datetime.datetime
        date1 = date1.strftime("%Y-%m-%d")  # str
        
        x.append(date)
        
        _x_monthly, _y_monthly = prepare_daily_avg_data(_db, date, date1, time_from, time_to)
        y.append(sum(_y_monthly) / len(_y_monthly))
        
        del _x_monthly, _y_monthly
        now = datetime.datetime(year = _year, month = _month, day = now.day)
        
    return x, y


def plot_single_night_2locations(_db1, _db2, datetime_from, datetime_to, label):
    # single-night short-term data
    x_db1, y_db1 = prepare_single_night_data(_db1, datetime_from, datetime_to)
    x_db2, y_db2 = prepare_single_night_data(_db2, datetime_from, datetime_to)

    # plot graph
    fig = Figure("Single-night NSB")
    fig.plot([datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in x_db1], y_db1, label = label[0])
    fig.plot([datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in x_db2], y_db2, label = label[1])
    
    # division_number = 5
    # division_length = int(len(x_AP) / division_number)
    fig.set_axes_properties(
            title = f"{datetime_from} to {datetime_to}",
            xlabel = "Datetime",
            ylabel = "NSB",
            # xticks = [i * division_length for i in range(division_number)] + [len(x_AP) - 1],
            # xticklabels = ["0", "100"],
            ylim = [min(y_db1), max(y_db1) + 1],
            ygrid = True,
            legend = True
    )
    fig.show()

    del x_db1, y_db1, x_db2, y_db2


def plot_daily_avg_2locations(_db1, _db2, date_range, time_range, label):
    # long-term data
    x_db1, y_db1 = prepare_daily_avg_data(_db1, date_range[0], date_range[1], time_range[0], time_range[1])
    x_db2, y_db2 = prepare_daily_avg_data(_db2, date_range[0], date_range[1], time_range[0], time_range[1])
    
    # plot graph
    fig = Figure("Daily Average NSB")
    fig.plot([datetime.datetime.strptime(i, "%Y-%m-%d") for i in x_db1], y_db1, label = label[0])
    fig.plot([datetime.datetime.strptime(i, "%Y-%m-%d") for i in x_db2], y_db2, label = label[1])
    
    # division = int(len(x) / 3)
    # xticks = [0, division, 2*division, len(x) - 1]
    fig.set_axes_properties(
            title = f"{date_range[0]} to {date_range[1]}",
            xlabel = "Date",
            ylabel = "NSB",
            # xticks = xticks,
            # xticklabels = [x[i] for i in xticks],
            ylim = [min(y_db1) - 1, max(y_db1) + 1],
            ygrid = True,
            legend = True
    )
    fig.show()
    
    del x_db1, y_db1, x_db2, y_db2


def plot_monthly_avg_2locations(_db1, _db2, date_range, time_range, label):
    x_db1, y_db1 = prepare_monthly_avg_data(_db1, date_range[0], date_range[1], time_range[0], time_range[1])
    x_db2, y_db2 = prepare_monthly_avg_data(_db2, date_range[0], date_range[1], time_range[0], time_range[1])
    
    # plot graph
    fig = Figure("Monthly Average NSB")
    fig.plot([datetime.datetime.strptime(i, "%Y-%m-%d") for i in x_db1], y_db1, label = label[0])
    fig.plot([datetime.datetime.strptime(i, "%Y-%m-%d") for i in x_db2], y_db2, label = label[1])
    
    fig.set_axes_properties(
            title = f"{date_range[0]} to {date_range[1]}",
            xlabel = "Date",
            ylabel = "NSB",
            # xticks = xticks,
            # xticklabels = [x[i] for i in xticks],
            ylim = [min(y_db1) - 1, max(y_db1) + 1],
            ygrid = True,
            legend = True
    )
    fig.show()
    
    del x_db1, y_db1, x_db2, y_db2



# load the dataset into memory
# todo memory inefficient
# db_AP = DataBase("iyl/sqmle_nsb_data_table/AP.csv")
# db_TST = DataBase("iyl/sqmle_nsb_data_table/TST.csv")
# db_HKU = DataBase("iyl/sqmle_nsb_data_table/HKU.csv")
db_ELO = DataBase("iyl/sqmle_nsb_data_table/ELO.csv")
db_PAOF = DataBase("iyl/sqmle_nsb_data_table/PAOF.csv")

plot_single_night_2locations(
        db_ELO,
        db_PAOF,
        "2020-06-28 18:00:00",
        "2020-06-29 07:00:00",
        ["ELO", "PAOF"]
)

plot_daily_avg_2locations(
        db_ELO,
        db_PAOF,
        ["2020-06-01", "2020-06-30"],
        ["20:00:00", "05:00:00"],
        ["ELO", "PAOF"]
)

plot_monthly_avg_2locations(  # memory inefficient & heavy
        db_ELO,
        db_PAOF,
        ["2020-01-01", "2021-01-01"],
        ["20:00:00", "05:00:00"],
        ["ELO", "PAOF"]
)

# free up memory
db_ELO.close()
db_PAOF.close()
"""

'''
# K-SHAPE
import os
import pandas as pd
import numpy as np
from tslearn import clustering  # .KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from figure import Figure

# STEP 1
print("[INFO] STEP 1 : Gather useful data")

def custom_collect_data():
    NSB = NightSkyBrightness(
            "iyl/sqmle_nsb_data_table/HKU.csv",
            "src/Sun/Sun_rise_set_2018-2026.csv",
            "src/Moon/Moon_rise_set_2018-2026.csv"
    )
    db_HKU_list = []

    for year in range(2022, 2026 + 1):
        _db_list = []

        for month in range(1, 12 + 1):
            # get number of days in a month
            month1 = datetime.date(year = year, month = month, day = 1)
            month2 = datetime.date(year = year if month + 1 <= 12 else year + 1, month = (month + 1) % 12 if month + 1 > 12 else month + 1, day = 1)
            days = month2 - month1
            days = days.days

            for day in range(1, days + 1):
                # print(f"[INFO] Reading data at {year}-{month:02}-{day:02}")
                df = NSB.get_one_night_data(f"{year}-{month:02}-{day:02}")[2]  # pandas.DataFrame

                # convert str datetime to Python datetime
                df["received_adjusted"] = pd.to_datetime(df["received_adjusted"])

                # separate datetime into 2 columns: date, time
                df["date"] = df["received_adjusted"].dt.date
                df["time"] = df["received_adjusted"].dt.time

                # append to data structure
                # db_HKU_list.append(df[["date", "time", "nsb"]])
                _db_list.append(df[["date", "time", "nsb"]])  # internal

                del df

        # export the yearly dataframe (just in case)
        _db = DataBase().concat(_db_list)
        DataBase().export(f"src/HKU_{year}.csv", _db)
        # append to data structure
        db_HKU_list.append(_db)

    # merge all gathered data into 1 dataframe
    db_HKU = DataBase().concat(db_HKU_list)  # pandas.DataFrame
    del db_HKU_list

    # export this dataframe for convenience future processing
    DataBase().export("src/HKU.csv", dataframe = db_HKU)

    return db_HKU


if os.path.exists("src/HKU.csv"):
    db_HKU = DataBase("src/HKU.csv").read()

else:
    db_HKU = custom_collect_data()


def custom_preprocess_data(db_HKU):
    # STEP 2
    print("[INFO] STEP 2 : Reshape into matrix")

    # drop duplicated row (2021-11-29 20:2X:XX 8.7)
    # print(db_HKU[db_HKU.duplicated()])
    print(f"[INFO] Drop duplicated rows: {db_HKU.duplicated().sum()}")
    # only look for duplicates in column "date" and "time", regardless "nsb"
    # keep first occurrence value for "nsb" for duplicated rows
    db_HKU = db_HKU.drop_duplicates(subset = ["date", "time"], keep = "first")

    # convert date after midnight to previous date --> match the same astronomical night
    _db_column_time = pd.to_datetime(db_HKU["date"] + " " + db_HKU["time"])  # convert whole column to datetime
    _filters = (
            (_db_column_time.dt.time >= pd.to_datetime("00:00").time()) &
            (_db_column_time.dt.time <= pd.to_datetime("08:00").time())
    )
    db_HKU["astronomical_date"] = db_HKU["date"]  # copy from calendar date
    # convert date after midnight to previous date in column "astronomical_date"
    db_HKU.loc[_filters, "astronomical_date"] = (_db_column_time[_filters] - pd.Timedelta(days = 1)).dt.strftime("%Y-%m-%d")
    # print(db_HKU.loc[db_HKU["astronomical_date"] == "2018-01-01", "nsb"])

    # Pivot Data: Reshape into a nightly curve matrix
    # [       time1 time2 ...
    #   date1 nsb11 nsb12 ...
    #   date2 nsb21 nsb22 ... ]
    db_HKU_matrix = db_HKU.pivot(
            index = "astronomical_date",
            columns = "time",
            values = "nsb"
    )
    del db_HKU

    # reorder matrix column "time" to match physical chronological order
    _matrix_column = db_HKU_matrix.columns
    _col_first = [i for i in _matrix_column if i >= "16:00:00"]
    _col_first.sort()
    _col_last = [i for i in _matrix_column if i <= "08:00:00"]
    _col_last.sort()

    if len(_col_first) + len(_col_last) == len(_matrix_column):
        db_HKU_matrix = db_HKU_matrix[_col_first + _col_last]

    else:
        raise ValueError("[ERROR] in reordering matrix columns")


    # STEP 3
    print("[INFO] STEP 3 : Preprocess data")

    data = []
    row_date = []
    drop_count = 0
    window_size = 1280  # standardize each night to fixed number of data points

    for date, row in db_HKU_matrix.iterrows():
        # loop each row (night) in matrix
        # find index for sun set/rise time
        non_nan_index = np.where(row.notna())[0]

        if len(non_nan_index) == 0:
            # no valid data point
            drop_count += 1
            continue

        # the exact useful night-time data
        _data = row.iloc[non_nan_index[0]:non_nan_index[-1] + 1]  # time with NSB

        # if _data.isna().mean() > 0.20:
        #     drop if less than 80% data
        #     continue

        # linear interpolate missing data
        # Parallelized C-level row interpolation
        _interpolated_data = _data.interpolate(method = "linear", limit_direction = "both")

        # resample columns (time of each night's data point) so that all nights share identical vector layout
        _resampled_data = np.interp(
                np.linspace(0, len(_interpolated_data) - 1, window_size),
                np.arange(len(_interpolated_data)),
                _interpolated_data.values
        )

        # append to data structure
        data.append(_resampled_data)
        row_date.append(date)

    print(f"[INFO] Dropped {drop_count} empty row")

    # reshape into 3D NumPy array for tslearn
    # (n_samples, n_timesteps, n_features)
    X = np.array(data)[:, :, np.newaxis]

    # export to CSV
    _df = pd.DataFrame(
            data = X.squeeze(axis = 2),
            index = row_date,
            columns = [f"time {i:03d}" for i in range(window_size)]
    )
    _df.index.name = "astronomical_date"
    _df.to_csv("src/HKU_interpolated.csv")
    print("[INFO] Exported src/HKU_interpolated.csv")

    return X, row_date, data, _df


if os.path.exists("src/HKU_cleaned_interpolated.csv"):
    db_HKU_interpolated = DataBase("src/HKU_cleaned_interpolated.csv").read(index_col = "astronomical_date")
    X = db_HKU_interpolated.values[:, :, np.newaxis]

else:
    X, *args, db_HKU_interpolated = custom_preprocess_data(db_HKU)


# STEP 4
print("[INFO] STEP 4 : Train the model")
# Z-normalize curves across time axes
# data is normalized to Z-score: how many standard deviation the point away from mean value
# higher Z-score = dimmer in our case
X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)

# sample dataset
print("[INFO] Preparing sample train dataset")
np.random.seed(42)
sample_size = min(365, X_scaled.shape[0])
print(f"[INFO] Train sample size : {sample_size}")
train_indices = np.random.choice(X_scaled.shape[0], size = sample_size, replace = False)
X_train = X_scaled[train_indices]

# train model on sample dataset
print("[INFO] Training K-shape model on train dataset")

def find_optimal_k(k_from, k_to):

    def pre_train_k_shape_model(train_data, k):
        np.random.seed(42)
        _kshape_model = clustering.KShape(
                n_clusters = k,
                random_state = 42,
                n_init = 5,  # number of initialization runs the algorithm will perform, select best result
                max_iter = 50
        )
        labels = _kshape_model.fit_predict(train_data)  # fit / fit_predict
        # Silhouette Score (Uses Shape-Based Distance metric)

        return _kshape_model.inertia_  #, clustering.silhouette_score(train_data, labels, metric = "dtw")  # sbd / dtw

    inertia = 1e3  # find min
    k_inertia = 0
    # silhouette_score = -1e3  # find max
    # k_silhouette_score = 0

    for k in range(k_from, k_to + 1):  # [15, 25]
        print(f"Trying number of cluster k = {k}")
        _inertia = pre_train_k_shape_model(X_train, k)

        if _inertia < inertia:
            inertia = _inertia
            k_inertia = k

        else:
            # number of k is optimized
            print("Found minimum inertia")
            break

    print(f"Number of cluster (min inertia) : {k_inertia}")
    # print(f"Number of cluster (max silhouette_score) : {k_silhouette_score}")

    return k_inertia  # , k_silhouette_score


if os.path.exists("K-Shape_model_HKU.json"):
    # load existing K-shape model
    print("[INFO] Found existing K-shape model. Loading . . .")
    kshape_model = clustering.KShape.from_json("K-Shape_model_HKU.json")
    print("[INFO] Loaded K-shape model")

else:
    # find optimize k
    optimal_k = find_optimal_k(20, 30)

    # train model
    print(f"[INFO] Training actual K-shape model (k = {optimal_k}) on train dataset")
    np.random.seed(42)
    kshape_model = clustering.KShape(
            n_clusters = optimal_k,  # 23
            random_state = 42,
            n_init = 5,
    )
    kshape_model.fit(X_train)

    # save the model
    kshape_model.to_json("K-Shape_model_HKU.json")
    print("[INFO] K-shape model saved as K-Shape_model_HKU.json")


# STEP 5
print("[INFO] STEP 5 : Evaluate data from model")
# assign cluster label to data point
print("[INFO] Predicting data's cluster ID")
X_label = kshape_model.predict(X_scaled)  # np 1D-array of cluster id for each night, label each night with cluster id

# get centroid
centroids = kshape_model.cluster_centers_  # 3D tensor, shape (n_clusters, target_length, 1)
# print( len(centroids) )  # number of clusters

# gather data from same cluster
print("[INFO] Gathering data from same cluster")
clusters_data_scaled = {}
cluster_data_raw = {}

for cluster_id in range(len(centroids)):
    clusters_data_scaled[cluster_id] = X_scaled[X_label == cluster_id]  # slicing on 3D tensor
    cluster_data_raw[cluster_id] = db_HKU_interpolated.values[X_label == cluster_id, :]  # slicing on 2D matrix

# visualize each cluster
print("[INFO] Plotting cluster profile")
fig_z_scaled = Figure("K-shape Clustering on NSB Data (2018-2026)", 4, 6)
fig_raw = Figure("K-shape Clustering on NSB Data (2018-2026)", 4, 6)

for i in range(len(centroids)):
    # plot Z-SCORE SCALED data belonging to this cluster
    _data = clusters_data_scaled[i]  # curves belonging to this cluster
    sample_size = min(150, len(_data))
    np.random.seed(42)
    sample_index = np.random.choice(len(_data), sample_size, replace = False)

    for data_index in sample_index:
        fig_z_scaled.plot(
                np.linspace(0, 1, len(centroids[i])),
                _data[data_index],
                axes_index = i,
                color = "gray",
                alpha = 0.15
        )

    # plot centroid
    fig_z_scaled.plot(
            np.linspace(0, 1, len(centroids[i])),
            centroids[i],
            axes_index = i,
            color = "crimson",
            lw = 1,
            label = f"Centroid {i}"
    )

    fig_z_scaled.set_axes_properties(
            axes_index = i,
            title = f"Cluster {i}",
            xlabel = "Time",
            ylabel = "Normalized Brightness\n(Z-score)",
            ylim = [-5, 5],
            xticks = [0, 0.5, 1],
            xticklabels = ["Sun set", "Midnight", "Sun rise"],
            invert_yaxis = True,
            legend = True
    )

    # plot RAW data belonging to this cluster
    _data = cluster_data_raw[i]  # curves belonging to this cluster

    # take mean on each curve in mag arcsec⁻² --> representative for the cluster
    # eliminate effect of small fluctuation / random noise
    # provides a statistically sound, representative absolute brightness value
    _data_mean = np.mean(_data, axis = 0)

    # take sample from raw data
    sample_size = min(150, len(_data))
    np.random.seed(42)
    sample_index = np.random.choice(len(_data), sample_size, replace = False)

    for data_index in sample_index:
        fig_raw.plot(
                np.linspace(0, 1, len(centroids[i])),
                _data[data_index],
                axes_index = i,
                color = "gray",
                alpha = 0.15
        )

    # plot mean representative
    fig_raw.plot(
            np.linspace(0, 1, len(centroids[i])),
            _data_mean,
            axes_index = i,
            color = "crimson",
            lw = 1,
            label = f"Mean {i}"
    )

    fig_raw.set_axes_properties(
            axes_index = i,
            title = f"Cluster {i}",
            xlabel = "Time",
            ylabel = "Brightness (mag arcsec⁻²)",
            ylim = [14, 19],
            xticks = [0, 0.5, 1],
            xticklabels = ["Sun set", "Midnight", "Sun rise"],
            invert_yaxis = True,
            legend = True
    )

fig_z_scaled.show()
fig_raw.show()

exit()


def calculate_moon_phase(dates):
    """
    Calculates lunar phase fraction (0.0 to 1.0) and assigns descriptive labels
    using a vectorized baseline method.
    """
    # Known historical New Moon baseline date
    baseline_new_moon = pd.to_datetime('2000-01-06')
    synodic_month = 29.530588853  # Precision length of a lunar cycle in days

    # Calculate exact decimal days elapsed since baseline
    delta_days = (dates - baseline_new_moon).dt.total_seconds() / (24 * 3600)

    # Extract the decimal remainder representing the current position in the cycle (0.0 - 1.0)
    phase_fraction = (delta_days % synodic_month) / synodic_month

    # Vectorized classification into the 4 primary moon states
    # (or you can expand this to 8 phases if you want Crescent/Gibbous resolution)
    conditions = [
        (phase_fraction < 0.06) | (phase_fraction > 0.94),  # New Moon window
        (phase_fraction >= 0.06) & (phase_fraction < 0.44),  # Waxing Moon
        (phase_fraction >= 0.44) & (phase_fraction <= 0.56),  # Full Moon window
        (phase_fraction > 0.56) & (phase_fraction <= 0.94)  # Waning Moon
    ]
    labels = ['New Moon', 'Waxing Moon', 'Full Moon', 'Waning Moon']

    # Instantly maps across all 36,000 rows using C-level optimizations
    return np.select(conditions, labels, default = 'Waxing Moon')


# dataframe for the cluster-labeled data
df_trends = pd.DataFrame({
    "Date": pd.to_datetime(db_HKU_interpolated.index),
    "Cluster": all_cluster_labels
})
df_trends["Year"] = df_trends["Date"].dt.year
df_trends["Moon_Phase"] = calculate_moon_phase(df_trends['Date'])

# Create the final decadal trend report
yearly_trend_pct = pd.crosstab(df_trends['Year'], df_trends['Cluster'], normalize='index') * 100
print("\n--- 10-YEAR TREND PERCENTAGE MATRIX ---")
print(yearly_trend_pct)


# --- CROSS-ANALYSIS: Factors Contributing to Clusters ---
# This allows you to check if 'Cluster 3' is entirely driven by Full Moons!
cluster_by_moon = pd.crosstab(df_trends['Cluster'], df_trends['Moon_Phase'], normalize='index') * 100

print("\n--- MOON PHASE INFLUENCE PER CLUSTER (%) ---")
print(cluster_by_moon)
'''

"""
# K-MEDOIDS
import os
import joblib
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

# STEP 1
print("[INFO] STEP 1 : Gather useful data")
db_HKU = DataBase("src/HKU.csv").read()

data_list = []
date_list = []

# gather useful data
for date, group in db_HKU.groupby("astronomical_date"):
    # skipped drop duplicates and NaN
    # skipped setting minimum sample size
    data_list.append(group[["nsb", "moon_illuminated_ratio"]].values)
    date_list.append(date)


# STEP 2
print("[INFO] Step 2 : Resample data into fixed length and make alignment")
# data structural alignment
data_points_per_night = 1280  # standardize to fixed number of data points per night
# X_aligned = TimeSeriesResampler(sz = length).fit_transform(data_list)  # shape: (#nights, #data point per night, features = 2 = NSB + moon)
X_aligned = []
data_list_aligned = []
date_list_aligned = []

for data, date in zip(data_list, date_list):

    # if len(data) < 100:
        # too few data point
        # drop entirely
        # continue

    # align light curve to fixed progress timeline [0, 1]
    original_timeline = np.linspace(0, 1, len(data))
    target_timeline = np.linspace(0, 1, data_points_per_night)

    f_nsb = interp1d(original_timeline, data[:, 0], kind = "linear")
    # f_moon = interp1d(original_timeline, data[:, 1], kind = "linear")

    aligned_nsb = f_nsb(target_timeline)
    # aligned_moon = f_moon(target_timeline)

    # aligned_night = np.column_stack([aligned_nsb, aligned_moon])

    # append to data structure
    # X_aligned.append(aligned_night.flatten())
    # data_list_aligned.append(aligned_night)
    X_aligned.append(aligned_nsb)
    data_list_aligned.append(data[:, 0])
    date_list_aligned.append(date)

X_aligned = np.array(X_aligned)

# STEP 4
print("[INFO] Step 4 : K-Medoids clustering")

if os.path.exists("K-Medoids_model_HKU.pkl"):
    print("[INFO] Found existing K-Medoids model. Loading . . .")
    kmedoids_model = joblib.load("K-Medoids_model_HKU.pkl")
    print("[INFO] Done")

else:
    optimal_k = 0
    score = 1e-3  # find max

    print("[INFO] Finding optimal value of k")

    for k in range(15, 30):
        print(f"Trying k = {k}")
        _kmedoids = KMedoids(
                n_clusters = k,
                metric = "euclidean",
                random_state = 42
        )
        _label = _kmedoids.fit_predict(X_aligned)

        _silhouette_score = silhouette_score(X_aligned, _label, metric = "euclidean")

        if _silhouette_score > score:
            score = _silhouette_score
            optimal_k = k

        else:
            print(f"Found optimal k = {optimal_k}")
            break

    # train actual K-Medoids model
    kmedoids_model = KMedoids(n_clusters = optimal_k, metric = "euclidean", random_state = 42)

    # save the model
    joblib.dump(kmedoids_model, "K-Medoids_model_HKU.pkl")

cluster_labels = kmedoids_model.fit_predict(X_aligned)
# get medoids --> take actual data from cluster as representative
medoid_index = kmedoids_model.medoid_indices_


# STEP 5
print("[INFO] Step 5 : Plot result")

fig = Figure("K-Medoids Clustering", 3, 6)

for cluster_id, medoid_index in enumerate(medoid_index):
    medoid_date = date_list[medoid_index]
    medoid_data = data_list_aligned[medoid_index]
    cluster_data = X_aligned[cluster_labels == cluster_id].tolist()
    index = np.random.choice(len(cluster_data), size = min(50, len(cluster_data)), replace = False)
    
    for i in index:
        fig.plot(
                np.linspace(0, 1, len(cluster_data[i])),
                cluster_data[i],
                axes_index = cluster_id,
                color = "grey",
                alpha = 0.1
        )
        
    fig.plot(
            np.linspace(0, 1, len(medoid_data)),
            medoid_data,
            axes_index = cluster_id,
            label = f"Medoid {cluster_id}",
            color = "red"
    )
    
    fig.set_axes_properties(
            axes_index = cluster_id,
            title = f"Cluster {cluster_id}",
            xlabel = "time",
            ylabel = "NSB",
            invert_yaxis = True,
            legend = True
    )

fig.show()
"""

"""
# CRLI @ Pypots (run out of computer memory)
import numpy as np
from pypots.clustering import CRLI
from pypots.nn.functional import calc_chs, calc_dbs
# Initialized PyPOTS Ecosystem configuration file C:\\Users\\PAUL\\.pypots\\config.ini successfully.


# STEP 1
print("[INFO] Step 1 : Load data")
db_HKU = DataBase("src/HKU.csv").read()

# STEP 2
print("[INFO] Step 2 : Data preprocessing")
# group each night's data by astronomical date
# map "time" into sequential step for varied sun set/rise time
# db_HKU["time_step"] = db_HKU.groupby("astronomical_date").cumcount()

# pivot data
# [       time1 time2 ...
#   date1 nsb11 nsb12
#   date2 nsb21 nsb22
#   ...                   ]
# db_HKU_pivot = db_HKU.pivot(
#         index = "astronomical_date",
#         columns = "time_step",
#         values = "nsb"
# )

# extract dimensions of parameters
n_samples = len(db_HKU.groupby("astronomical_date"))  # db_HKU_pivot.shape[0]  # number of nights
n_steps = db_HKU.groupby("astronomical_date").size().max()  # db_HKU_pivot.shape[1]    # number of data points per night
n_features = 1  # night-sky brightness

# reshape into PyPOTS 3D array format: (n_samples, n_steps, n_features)
max_steps = db_HKU.groupby("astronomical_date").size().max()

data = []

for date, group in db_HKU.groupby("astronomical_date"):
    night_data = group.values[:, 2]  # all rows, 3rd column
    # night_data = group.iloc[:, 2].values.astype(np.float32)  # all rows, 3rd column

    if len(night_data) < 50:
        # drop night with insufficient data points
        continue

    if len(night_data) < max_steps:
        # length of data points < required
        padding = np.full(max_steps - len(night_data), np.nan, dtype = np.float32)
        night_data = np.concatenate([night_data, padding])  # append NaN at end of raw data

    data.append(night_data)

# Convert to the final 3D NumPy array for CRLI
X = np.array(data, dtype = np.float32).reshape(len(data), max_steps, 1)
# X = db_HKU_pivot.values.reshape((n_samples, n_steps, n_features))
train_indices = np.random.choice(X.shape[0], size = 365, replace = False)
X_train = X[train_indices]
train_dataset = {
    "X": X_train
}
dataset = {
    "X": X
}

# STEP 4
print("[INFO] Step 4 : Train CRLI model")

print("[INFO] Finding optimal number of cluster k")
chs_score = 1e-3  # find max
dbs_score = 1e3  # find min
optimal_k_chs = 0
optimal_k_dbs = 0
optimal_k = 0

for k in range(3, 30):
    print(f"Trying k = {k}")
    model = CRLI(
            n_steps = n_steps,
            n_features = n_features,
            n_clusters = k,
            epochs = 5,
            n_generator_layers = 2,
            rnn_hidden_size = 32,
            rnn_cell_type = "LSTM",  # GRU / LSTM
            batch_size = 16,  # lower batch size to prevent memory overloading for large dataset
            # patience = 5,  # Early stopping patience
            device = "cpu",
    )
    model.fit(train_set = train_dataset)
    predictions = model.predict(train_dataset)
    labels = predictions["clustering"]
    imputed_features = predictions["imputed_X"].reshape(n_samples, -1)
    _chs_score = calc_chs(imputed_features, labels)
    _dbs_score = calc_dbs(imputed_features, labels)

    if _chs_score > chs_score:
        chs_score = _chs_score
        optimal_k_chs = k

    if _dbs_score < dbs_score:
        dbs_score = _dbs_score
        optimal_k_dbs = k

    if k > max(optimal_k_chs, optimal_k_dbs):
        optimal_k = k - 1

    else:
        print(f"Found optimal k = {optimal_k}")
        break


crli_model = CRLI(
        n_steps = n_steps,
        n_features = n_features,
        n_clusters = optimal_k,
        n_generator_layers = 2,
        rnn_hidden_size = 32,
        rnn_cell_type = "GRU",  # GRU / LSTM
        batch_size = 32,
        epochs = 50,  # Increase for better convergence on real 8-year dataset
        patience = 10  # Early stopping patience
)
crli_model.fit(train_set = train_dataset)

result = crli_model.predict(dataset)
label = result["clustering"]  # cluster ID corresponding to db_HKU_pivot.index

# STEP 5
print("[INFO] Step 5 : Plot result")
print(label)
"""
