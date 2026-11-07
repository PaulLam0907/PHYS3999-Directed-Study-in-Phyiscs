"""
A module for night-sky brightness (NSB) analysis

1. Slice data by time windows
2. Algorithm :
    - K-Medoids (PAM) (+FPCA) (force data into a cluster, cannot find outlier, uniform cluster density, require guessing k)
    - K-Mean (mean is sensitive to extreme value in the cluster)
    - DBSCAN (require guessing radius of cluster)
    - HDBSCAN (can find outlier, group by density not by curve shape if implement feature, can implement feature e.g. moon illuminated ratio, do not require guessing number of cluster or cluster radius)
    - K-Shape (Z-score scaling don't preserve unit)
    - CRLI
    - X-Mean / G-Mean
    - Agglomerative Hierarchical Clustering (Ward’s Linkage)
    - Gaussian Mixture Models (GMM)
3. map each night to cluster class
• The Yearly Composition Shift: Count how many nights belonged to each cluster for Year 1 vs. Year 10. If Year 1 had 60% "Deep Dark Nights" (Cluster 1) and Year 10 only has 20%, you have successfully isolated a clear, quantifiable light pollution trend.
• The Transition Matrix: Calculate the probability of moving from one cluster type to another over the years, showing if the night sky is degrading or stabilizing.
4. distance metrics: DTW distance / cross-correlation (Shape-Based Distance (SBD))

Package Requirement :
    numpy
    pandas
    scikit-learn
    scikit-learn-extra
    pypots
"""
import os
import joblib
import h5py
import datetime
import numpy as np
from scipy import signal
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn_extra.cluster import KMedoids
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from pypots.clustering import CRLI
from pypots.nn.functional import calc_chs, calc_dbs

from database import DataBase


class NightSkyBrightness:
    
    def __init__(self,
                 night_sky_brightness_db = "data.csv",
                 sun_rise_set_db = "sun.csv",
                 moon_rise_set_db = "moon.csv"):
        """
        A class for basic statistical night_sky brightness (NSB) analysis
        
        :param night_sky_brightness_db: str, file name of NSB database (with extension), assumed csv
        :param sun_rise_set_db: str, file name of Sun rise/set database (with extension), assumed csv
        :param moon_rise_set_db: str, file name of Moon rise/set database (with extension), assumed csv
        """
        self.nsb_db = DataBase(night_sky_brightness_db)
        self.sun_db = DataBase(sun_rise_set_db)
        self.moon_db = DataBase(moon_rise_set_db)
        
        
    def _get_sun_moon_rise_set_time(self, sun_moon, rise_set, date):
        """
        Get Sun/Moon rise/set time
        Adjusted +75 minutes from official sun set time
        Adjusted -75 minutes from official sun rise time
        
        :param sun_moon: str "sun" or "moon"
        :param rise_set: str "rise" or "set" or "transition"
        :param date: str, YYYY-MM-DD, accept YYYY-MM-DD HH:MM:SS
        :return: str of HH:MM, nan if blank data point
        """
        # check to use which db
        if sun_moon.lower() == "sun":
            _db = self.sun_db
        
        elif sun_moon.lower() == "moon":
            _db = self.moon_db
        
        else:
            raise ValueError("sun_moon must be either \"sun\" or \"moon\" ")
        
        # check column in db
        rise_set = rise_set.upper()
        
        if not rise_set in _db.get_header():
            raise KeyError(f"{rise_set} not in database header (column name)")
        
        # get value from db
        result = _db.get_column(
                [rise_set],
                [
                    {"YYYY-MM-DD": ["=", date.split(" ")[0]]}
                ]
        )  # pandas.Dataframe
        
        if result.empty:
            # empty dataframe
            return np.nan
        
        result = result[rise_set].values[0]  # str
        
        if isinstance(result, float):
            
            if np.isnan(result):
                # blank data point / NaN
                return result
        
        result = [int(i) for i in result.split(":")]  # list [hour, minute]
        
        # adjust time to +-75 minutes
        time = datetime.time(hour = result[0], minute = result[1])
        adjusted_time = datetime.datetime.combine(datetime.datetime.today(), time)
        
        if sun_moon.lower() == "sun":
            
            if rise_set.upper() == "SET":
                adjusted_time += datetime.timedelta(minutes = 75)
            
            elif rise_set.upper() == "RISE":
                adjusted_time -= datetime.timedelta(minutes = 75)
        
        # convert to str
        result = adjusted_time.time()
        result = result.strftime("%H:%M")
        
        return result
    
    
    def get_sun_set_time(self, date):
        """
        Get Sun set time
        Adjusted +75 minutes from official sun set time
        
        :param date: str, YYYY-MM-DD, accept YYYY-MM-DD HH:MM:SS
        :return: str of HH:MM
        """
        # # get value from db
        # result = self.sun_db.get_column(
        #         ["SET"],
        #         [
        #             {"YYYY-MM-DD": ["=", date.split(" ")[0]]}
        #         ]
        # )  # pandas.Dataframe
        # result = result["SET"].values[0]  # str
        # result = [int(i) for i in result.split(":")]  # list [hour, minute]
        #
        # # adjust time to +75 minutes
        # sun_set_time = datetime.time(hour = result[0], minute = result[1])
        # adjusted_sun_set_time = datetime.datetime.combine(datetime.datetime.today(), sun_set_time)
        # adjusted_sun_set_time += datetime.timedelta(minutes = 75)
        #
        # # convert to str
        # result = adjusted_sun_set_time.time()
        # result = result.strftime("%H:%M")
        
        return self._get_sun_moon_rise_set_time("sun", "set", date)
    
    
    def get_sun_rise_time(self, date):
        """
        Get Sun rise time
        Adjusted -75 minutes from official sun set time

        :param date: str, YYYY-MM-DD, accept YYYY-MM-DD HH:MM:SS
        :return: str of HH:MM
        """
        return self._get_sun_moon_rise_set_time("sun", "rise", date)
    
    
    def get_moon_set_time(self, date):
        """
        Get Moon set time

        :param date: str, YYYY-MM-DD, accept YYYY-MM-DD HH:MM:SS
        :return: str of HH:MM
        """
        return self._get_sun_moon_rise_set_time("moon", "set", date)
    
    
    def get_moon_transition_time(self, date):
        """
        Get Moon transition time
        
        :param date: str, YYYY-MM-DD, accept YYYY-MM-DD HH:MM:SS
        :return: str of HH:MM
        """
        return self._get_sun_moon_rise_set_time("moon", "tran.", date)
    
    
    def get_moon_rise_time(self, date):
        """
        Get Moon rise time

        :param date: str, YYYY-MM-DD, accept YYYY-MM-DD HH:MM:SS
        :return: str of HH:MM
        """
        return self._get_sun_moon_rise_set_time("moon", "rise", date)
    
    
    def get_moon_phase(self, date):
        """
        Use modular arithmetic to calculate approximate Moon phase
        synodic lunar month (29.53059 days)
        
        Google search : https://www.google.com.hk/search?q=mathmethod+to+calculate+mon+phase&sca_esv=3d4a812c9ccd5f2a&sxsrf=APpeQnsr7jzdYAag-M7fCm9TzWpRcmzrng%3A1791277095995&ei=J7jEar-nPOmy0-kP54uZ8A8&biw=1018&bih=497&uact=5&oq=mathmethod+to+calculate+mon+phase&gs_lp=Egxnd3Mtd2l6LXNlcnAiIW1hdGhtZXRob2QgdG8gY2FsY3VsYXRlIG1vbiBwaGFzZTIKECEYChigARjDBEiCF1DrEViDFnAFeAGQAQCYAYsBoAHHAqoBAzMuMbgBA8gBAPgBAZgCCaACgwPCAgoQABhHGNYEGLADwgINEAAYRxjWBBjJAxiwA8ICDhAAGIAEGIoFGJIDGLADwgIIEAAYgAQYogTCAggQABiJBRiiBMICBRAAGO8FmAMA4gMFEgExIECIBgGQBgaSBwM4LjGgB4wOsgcDMy4xuAfqAsIHBTAuNS40yAccgAgB&sclient=gws-wiz-serp
        https://en.wikipedia.org/wiki/Lunar_phase
        https://en.minkukel.com/various/calculating-moon-phase/
        https://share.google/SMLrurC7sYLh396Yi
        https://www.youtube.com/watch?v=zd_tELwFRk4&t=18s
        
        :param date: str, YYYY-MM-DD, accept YYYY-MM-DD HH:MM:SS
        :return: tuple (str: moon phase, float: illuminated ratio)
        """
        # known new moon datetime T0
        # 6 January 2000
        # 18:14 UTC + 8 = HKT
        new_moon_datetime = datetime.datetime(year = 2000, month = 1, day = 6, hour = 18, minute = 14)  # UTC
        new_moon_datetime += datetime.timedelta(hours = 8)  # HKT
        
        # target date T
        target_datetime = datetime.datetime.strptime(
                date,
                "%Y-%m-%d" if len(date.split(" ")) == 1 else "%Y-%m-%d %H:%M:%S"
        )
        
        # difference in day d = T - T0
        delta_t = target_datetime - new_moon_datetime
        delta_t = delta_t.total_seconds() / 24 / 60 / 60  # float
        
        # current Moon age A [0, 29.53] in cycle
        moon_age = delta_t % 29.53059
        
        # map of Moon phase
        moon_phase = {
            "New Moon": [0.0, 1.84],
            "Waxing Crescent": [1.84, 5.54],
            "First Quarter": [5.54, 9.22],
            "Waxing Gibbous": [9.22, 12.91],
            "Full Moon": [12.91, 16.61],
            "Waning Gibbous": [16.61, 20.30],
            "Third Quarter": [20.30, 23.99],
            "Waning Crescent": [23.99, 27.68],
            "New Moon ": [27.68, 29.53]
        }
        
        # get target Moon phase
        target_moon_phase = ""
        
        for moon_phase_day in moon_phase.items():
            
            if moon_age < moon_phase_day[1][1]:
                target_moon_phase = moon_phase_day[0].strip()
                break
        
        # illumination percentage
        # visible illuminated fraction k
        # k = 0: new moon
        # k = 1: full moon
        illuminated_ratio = (1 - np.cos(2 * np.pi * moon_age / 29.53059)) / 2
        
        return target_moon_phase, illuminated_ratio
    
    
    def get_one_night_data(self, date_night_from, datetime_column_name = "received_adjusted", nsb_column_name = "nsb"):
        """
        Prepare one night data of NSB with Sun set/rise time adjusted
        Retrieve data from [the night of date_night_from] to [the dawn of next day of date_night_from]
        
        :param date_night_from: str, YYYY-MM-DD, accept YYYY-MM-DD HH:MM:SS
        :param datetime_column_name: str (optional), column name for date & time in dataset
        :param nsb_column_name: str (optional), column name for NSB in dataset
        :return: tuple (pandas.Series: datetime, pandas.Series: NSB, pandas.DataFrame)
        """
        # prepare datetime range
        time_from = self.get_sun_set_time(date_night_from)  # str, Sun set time
        datetime_from = date_night_from.split(" ")[0] + " " + time_from  # str
        
        date_to = datetime.datetime.strptime(
                date_night_from,
                "%Y-%m-%d" if len(date_night_from.split(" ")) == 1 else "%Y-%m-%d %H:%M:%S"
        )  # datetime
        date_to += datetime.timedelta(days = 1)  # add day by 1
        date_to = date_to.strftime("%Y-%m-%d")  # str
        
        time_to = self.get_sun_rise_time(date_to)  # str, Sun rise time
        if isinstance(time_to, float):
            if np.isnan(time_to):
                time_to = "00:00"  # todo fix December 31
        datetime_to = date_to + " " + time_to  # str
        
        # get data from db
        print(f"[INFO] Loading data ({datetime_from} -> {datetime_to}) from database")
        data = self.nsb_db.get_column(
                [datetime_column_name, nsb_column_name],
                filters = [
                    {datetime_column_name: [">=", datetime_from]},
                    {datetime_column_name: ["<=", datetime_to]},
                    # {"nsb": ["<", 23]},
                    # {"nsb": [">", 10]}
                ]
        )
        print("[INFO] Data loaded to memory")
        
        x = data[datetime_column_name]  # date & time
        y = data[nsb_column_name]  # NSB
        
        # del data
        return x, y, data
    
    
    def one_night_average(self, date_night_from):
        """
        Calculate average of NSB in one night
        night = from [the night of date_night_from] to [the dawn of next day of date_night_from]
        
        :param date_night_from: str, YYYY-MM-DD, accept YYYY-MM-DD HH:MM:SS
        :return: float
        """
        data_nsb = self.get_one_night_data(date_night_from)[1]
        print(f"[INFO] Calculating {date_night_from.split(' ')[0]} average NSB")
        average = sum(data_nsb) / len(data_nsb) if len(data_nsb) else np.nan
        print("[INFO] Done")
        
        return average
    
    
    def one_month_average(self, date):
        """
        Calculate average of NSB in one month, by taking average of each day's NSB
        one month = from day 1's night to last day's night of the month
        
        :param date: str, YYYY-MM-DD, accept YYYY-MM-DD HH:MM:SS
        :return: tuple (float: month average, list: daily average, list: moon phase)
        """
        date = datetime.datetime.strptime(
                date,
                "%Y-%m-%d" if len(date.split(" ")) == 1 else "%Y-%m-%d %H:%M:%S"
        )  # datetime.datetime
        date = datetime.datetime(year = date.year, month = date.month, day = 1)  # set to day 1 of the month
        now = date  # datetime.datetime
        data = []  # one-night average in the month
        moon_phase = []
        
        # prepare one-night average NSB in 1 month
        while now.month == date.month:
            night_average = self.one_night_average(now.strftime("%Y-%m-%d"))  # get one night average NSB
            data.append(night_average)  # append to data structure
            moon_phase.append(self.get_moon_phase(now.strftime("%Y-%m-%d")))  # append Moon phase to data structure
            now += datetime.timedelta(days = 1)  # continue to next day
        
        # take average
        print(f"[INFO] Calculating {date.year}-{date.month} average NSB")
        # result = sum(data) / len(data)
        result = np.nanmean(np.array(data))
        print("[INFO] Done")
        
        return result, data, moon_phase
    
    
    def one_year_average(self, date):
        """
        Calculate average of NSB in one year, by taking average of each month's average NSB
        one year = from January to December of the interested year
        
        :param date: str, YYYY-MM-DD, accept YYYY-MM-DD HH:MM:SS
        :return: tuple (float: year average, list: monthly average)
        """
        year = date.split(" ")[0][0:4]  # str
        data = []  # one-month average in the year
        
        # prepare one-month average NSB in 1 year
        for month in range(1, 12 + 1):
            month_average = self.one_month_average(f"{year}-{month:02}-01")[0]  # get one month average NSB
            data.append(month_average)  # append to data structure
        
        # take average
        print(f"[INFO] Calculating {year} average NSB")
        # result = sum(data) / len(data)
        result = np.nanmean(np.array(data))
        print("[INFO] Done")
        
        return result, data
    

class Clustering:
    
    def __init__(self):
        self.DATASET = None
        self.TRAIN_SET = None
        self.VALIDATION_SET = None
        self.TEST_SET = None
        self.TRAIN_METADATA = None
        self.VALIDATION_METADATA = None
        self.TEST_METADATA = None
        self.MODEL = None
    
    
    def dataset(self, data_set, train_set_ratio, test_set_ratio, random_state = 42):
        """
        Use sklearn.model_selection.train_test_split() to split dataset
        into [train, test, validation set]

        :param data_set: the raw dataset to split
        :param train_set_ratio: float [0, 1], ratio of the train set to whole dataset
        :param test_set_ratio: float [0, 1], ratio of the test set to whole dataset
        :param random_state: int
        :return: tuple (train set, test set, validation set)
        """
        # split the dataset into different portion for different purpose
        # test set = unseen data
        _, test_set = train_test_split(data_set, test_size = test_set_ratio, random_state = random_state)
        # (1 - test_set_ratio) for both validation_set and train_set
        # test_set_ratio  = test_set / all  ( given )
        # train_set_ratio = train_set / all  ( given )
        # let x = ratio of train_set to [train_set + validation_set] = train_set / (train_set + validation_set)
        # x * (train_set + validation_set) = train_set_ratio * all
        # => x = train_set_ratio * all / (train_set + validation_set)
        # => x = train_set_ratio / (1 - test_set_ratio)
        validation_set, train_set = train_test_split(_, test_size = train_set_ratio / (1 - test_set_ratio),
                                                     random_state = random_state)
        
        # store it into class properties
        self.DATASET = data_set
        self.TRAIN_SET = train_set
        self.VALIDATION_SET = validation_set
        self.TEST_SET = test_set
        
        # log
        print(f"[INFO] Train set ({train_set.shape[0] / data_set.shape[0] * 100:.03}%) : {train_set.shape}")
        print(f"[INFO] Test set ({test_set.shape[0] / data_set.shape[0] * 100:.03}%) : {test_set.shape}")
        print(f"[INFO] Validation set ({validation_set.shape[0] / data_set.shape[0] * 100:.03}%) : {validation_set.shape}")
        
        return train_set, test_set, validation_set
    
    
    def dataset_with_metadata(self, data_set, meta_data, train_set_ratio, test_set_ratio, random_state = 42):
        """
        Use sklearn.model_selection.train_test_split() to split dataset
        into [train, test, validation set]
        This method can be treated as the "extension" of dataset()

        :param data_set: the raw dataset to split
        :param meta_data: dict, metadata associated with each data points in dataset
        :param train_set_ratio: float [0, 1], ratio of the train set to whole dataset
        :param test_set_ratio: float [0, 1], ratio of the test set to whole dataset
        :param random_state: int
        :return: dict
        {
            "train": {
                "data" : ...,
                <key of metadata>: ...
            },
            "test": {
                "data" : ...,
                <key of metadata>: ...
            },
            "validation": {
                "data" : ...,
                <key of metadata>: ...
            }
        }
        """
        metadata_key = list(meta_data.keys())
        result = {
            "train": {},
            "test": {},
            "validation": {}
        }
        
        # test set = unseen data
        temp_metadata = []
        _ = train_test_split(
                data_set,
                *[meta_data[key] for key in metadata_key],
                test_size = test_set_ratio,
                random_state = random_state
        )
        result["test"]["data"] = _[1]  # test set
        
        # store metadata
        for i in range(1, len(metadata_key) + 1):
            temp_metadata.append(_[2 * i])  # non-test metadata
            result["test"][metadata_key[i - 1]] = _[2 * i + 1]  # test metadata
            self.TEST_METADATA = _[2 * i + 1]  # test metadata
        
        # (1 - test_set_ratio) for both validation_set and train_set
        __ = train_test_split(
                _[0],  # non-test set data
                *temp_metadata,  # non-test set metadata
                test_size = train_set_ratio / (1 - test_set_ratio),
                random_state = random_state
        )
        result["validation"]["data"] = __[0]  # validation set
        result["train"]["data"] = __[1]  # train set
        
        # store metadata
        for i in range(1, len(metadata_key)):
            result["validation"][metadata_key[i - 1]] = __[2 * i]  # validation metadata
            self.VALIDATION_METADATA = __[2 * i]  # validation metadata
            result["train"][metadata_key[i - 1]] = __[2 * i + 1]  # train metadata
            self.TRAIN_METADATA = __[2 * i + 1]  # train metadata
        
        # store data to class property
        self.DATASET = data_set
        self.TRAIN_SET = result["train"]["data"]
        self.VALIDATION_SET = result["validation"]["data"]
        self.TEST_SET = result["test"]["data"]
        
        # log
        print(f"[INFO] Train set count : {len(self.TRAIN_SET)} ({len(self.TRAIN_SET) / len(data_set) * 100:.03}%)")
        print(f"[INFO] Test set count : {len(self.TEST_SET)} ({len(self.TEST_SET) / len(data_set) * 100:.03}%)")
        print(f"[INFO] Validation set count : {len(self.VALIDATION_SET)} ({len(self.VALIDATION_SET) / len(data_set) * 100:.03}%)")
        
        return result
    
    
    def train_test_validation_set(self, train_set, test_set, validation_set):
        """
        Store custom train, test, validation dataset into class property

        :param train_set: seen dataset for training
        :param test_set: unseen dataset for testing
        :param validation_set:
        :return: None
        """
        self.TRAIN_SET = train_set
        self.TEST_SET = test_set
        self.VALIDATION_SET = validation_set
    
    
    def load_model(self, file_name):
        """
        Load existing ML model and store to class property

        :param file_name: str, file name of the saved model (with extension)
        :return: the loaded model
        """
        model = joblib.load(file_name)
        self.MODEL = model
        
        return model
    

class KMedoid(Clustering):
    
    def __init__(self):
        """
        A class for K-Medoids model clustering
        
        Before working on the model,
        user should call either dataset() or train_test_validation_set() or dataset_with_metadata()
        to provide data for model training and evaluation using class methods
        """
        # Clustering.__init__(self)
        super().__init__()
        
    
    def find_optimal_k(self, k_range, train_set = None, validation_set = None, metric = "euclidean", random_state = 42):
        """
        Find optimal number of cluster k
        (hyperparameter tuning)
        
        :param k_range: tuple of int e.g. (3, 10), range of k to test
        :param train_set: train dataset for model training. Default to self.TRAIN_SET
        :param validation_set: validation set. Default to self.VALIDATION_SET
        :param metric: str, distance metric for model e.g. euclidean, correlation, manhattan
        :param random_state: int
        :return: tuple (int: optimal k, float: silhouette score)
        """
        # default to class property if train_set and validation_set are not provided
        if not train_set:
            train_set = self.TRAIN_SET
            
        if not validation_set:
            validation_set = self.VALIDATION_SET
        
        # find optimal number of cluster k
        optimal_silhouette_k = 2
        optimal_dbi_k = 2
        silhouette_score_ = -1  # silhouette score find max, greatest separation between clusters
        inertia = []  # Elbow Method
        davies_bouldin_score_ = 1e3  # find min
        
        print("[INFO] Finding optimal value of k")
        print(f"k  |  Silhouette Score  |  Davies Bouldin (DBI) Score  |  Inertia")
        
        for k in range(k_range[0], k_range[1] + 1):
            np.random.seed(random_state)
            _model = KMedoids(
                    n_clusters = k,
                    metric = metric,
                    # method = "alternate",
                    random_state = random_state
            )
            _model.fit(train_set)  # train model on train_set
            _label = _model.predict(validation_set)  # predict data on validation_set
            
            # evaluate how well the validation data fit into trained clusters
            _silhouette_score = silhouette_score(validation_set, _label, metric = metric)
            _davies_bouldin_score = davies_bouldin_score(validation_set, _label)
            print(f"{k} | {_silhouette_score} | {_davies_bouldin_score} | {_model.inertia_}")
            
            if _silhouette_score > silhouette_score_:
                silhouette_score_ = _silhouette_score
                optimal_silhouette_k = k
            
            if _davies_bouldin_score < davies_bouldin_score_:
                davies_bouldin_score_ = _davies_bouldin_score
                optimal_dbi_k = k
                
            # else:
            #     print(f"Found optimal k = {optimal_k}")
            #     break
        
        print(f"Optimal k for Silhouette Score : {optimal_silhouette_k}")
        print(f"Optimal k for Davies Bouldin (DBI) Score : {optimal_dbi_k}")
        
        return optimal_silhouette_k, silhouette_score_  # todo
    
    
    def fit(self, k, train_set = None, metric = "euclidean", random_state = 42, save_model = True):
        """
        Train K-Medoids model
        
        :param k: int, number of clusters
        :param train_set: train dataset for model training. Default to self.TRAIN_SET
        :param metric: str, distance metric for model
        :param random_state: int
        :param save_model: bool, indicate whether to save the trained model or not
        :return: the trained K-Medoids model
        """
        # default to class property if train_set are not provided
        if not train_set:
            # train_set = self.TRAIN_SET
            train_set = np.vstack((self.TRAIN_SET, self.VALIDATION_SET))
            
        # train the model
        print(f"[INFO] Training K-Medoids model with k = {k} clusters")
        model = KMedoids(
                n_clusters = k,
                metric = metric,
                random_state = random_state
        )
        model.fit(train_set)
        print("[INFO] Done")
        
        # save the model to class property
        self.MODEL = model
        print("[INFO] You may find the trained model in self.MODEL")
        
        # save the model to disk
        if save_model:
            joblib.dump(model, "K-Medoids_model.pkl")
            print("[INFO] Model saved as ./K-Medoids_model.pkl")
        
        return model
    
    
    def predict(self, model = None, test_set = None, metric = "euclidean"):
        """
        Model evaluation on unseen test set ( after fit() )
        with fixed cluster center
        
        :param model: the K-Medoids model. Default to self.MODEL
        :param test_set: test dataset for model evaluation. Default to self.TEST_SET
        :param metric: str, distance metric for model
        :return: tuple (cluster labels, silhouette score)
        """
        # default to class property if model and test_set are not provided
        if not model:
            model = self.MODEL
            
        if test_set is None:
            test_set = self.TEST_SET
        
        test_labels = model.predict(test_set)
        silhouette_score_ = silhouette_score(test_set, test_labels, metric = metric)
        print(f"[INFO] Test Set Silhouette Score: {silhouette_score_:.4f}")
        
        return test_labels, silhouette_score_


class Agglomerative(Clustering):
    
    def __init__(self):
        """
        A class for agglomerative clustering (bottom up)

        Before working on the model,
        user should call either dataset() or train_test_validation_set() or dataset_with_metadata()
        to provide data for model training and evaluation using class methods
        """
        # Clustering.__init__(self)
        super().__init__()
    
    
    def find_optimal_n(self, n_range, train_set = None, validation_set = None, metric = "euclidean", linkage = "ward"):
        """
        Find optimal number of cluster n
        (hyperparameter tuning)

        :param n_range: tuple of int e.g. (3, 10), range of n to test
        :param train_set: train dataset for model training. Default to self.TRAIN_SET
        :param validation_set: validation set. Default to self.VALIDATION_SET
        :param metric: str, distance metric for model e.g. euclidean, correlation, manhattan
        :param linkage: str, linkage method
        :return: tuple (int: optimal k, float: silhouette score)
        """
        # default to class property if train_set and validation_set are not provided
        if not train_set:
            train_set = self.TRAIN_SET
        
        if not validation_set:
            validation_set = self.VALIDATION_SET
        
        # find optimal number of cluster k
        optimal_silhouette_n = 2
        optimal_dbi_n = 2
        silhouette_score_ = -1  # silhouette score find max, greatest separation between clusters
        davies_bouldin_score_ = 1e3  # find min
        
        print("[INFO] Finding optimal value of k")
        print(f"n  |  Silhouette Score  |  Davies Bouldin (DBI) Score")
        
        for n in range(n_range[0], n_range[1] + 1):
            _model = AgglomerativeClustering(
                    n_clusters = n,
                    metric = metric,
                    linkage = linkage
            )
            _model.fit(train_set)  # train model on train_set
            
            _classifier = KNeighborsClassifier(n_neighbors = 1)  # todo KNN classifier
            _classifier.fit(train_set, _model.labels_)
            
            _label = _classifier.predict(validation_set)  # predict data on validation_set
            
            # evaluate how well the validation data fit into trained clusters
            _silhouette_score = silhouette_score(validation_set, _label, metric = metric)
            _davies_bouldin_score = davies_bouldin_score(validation_set, _label)
            print(f"{n} | {_silhouette_score} | {_davies_bouldin_score}")
            
            if _silhouette_score > silhouette_score_:
                silhouette_score_ = _silhouette_score
                optimal_silhouette_n = n
            
            if _davies_bouldin_score < davies_bouldin_score_:
                davies_bouldin_score_ = _davies_bouldin_score
                optimal_dbi_n = n
            
            # else:
            #     print(f"Found optimal k = {optimal_k}")
            #     break
        
        print(f"Optimal n for Silhouette Score : {optimal_silhouette_n}")
        print(f"Optimal n for Davies Bouldin (DBI) Score : {optimal_dbi_n}")
        
        return optimal_silhouette_n, silhouette_score_  # todo
    
    
    def fit(self, n_clusters, train_set = None, metric = "euclidean", linkage = "ward", save_model = True):
        """
        Train AgglomerativeClustering model

        :param n_clusters: int, number of clusters
        :param train_set: train dataset for model training. Default to self.TRAIN_SET
        :param metric: str, distance metric for model
        :param linkage: str, linkage method
        :param save_model: bool, indicate whether to save the trained model or not
        :return: the trained AgglomerativeClustering model
        """
        # default to class property if train_set are not provided
        if not train_set:
            # train_set = self.TRAIN_SET
            train_set = np.vstack((self.TRAIN_SET, self.VALIDATION_SET))
        
        # train the model
        print(f"[INFO] Training AgglomerativeClustering model with n_clusters = {n_clusters} clusters")
        model = AgglomerativeClustering(
                n_clusters = n_clusters,
                metric = metric,
                linkage = linkage
        )
        model.fit(train_set)
        print("[INFO] Done")
        
        # save the model to class property
        self.MODEL = model
        print("[INFO] You may find the trained model in self.MODEL")
        
        # save the model to disk
        if save_model:
            joblib.dump(model, "AgglomerativeClustering_model.joblib")
            print("[INFO] Model saved as ./AgglomerativeClustering_model.joblib")
        
        return model
    
    
    def predict(self, model = None, test_set = None, metric = "euclidean"):
        """
        Model evaluation on unseen test set ( after fit() )
        with fixed cluster center
        
        :param model: the AgglomerativeClustering model. Default to self.MODEL
        :param test_set: test dataset for model evaluation. Default to self.TEST_SET
        :param metric: str, distance metric for model
        :return: tuple (cluster labels, silhouette score)
        """
        # default to class property if model and test_set are not provided
        if not model:
            model = self.MODEL
        
        if test_set is None:
            test_set = self.TEST_SET
        
        test_labels = model["classifier"].predict(test_set)  # todo agglomerative evaluate test set
        silhouette_score_ = silhouette_score(test_set, test_labels, metric = metric)
        print(f"[INFO] Test Set Silhouette Score: {silhouette_score_:.4f}")
        
        return test_labels, silhouette_score_
    
    
    def fit_predict(self, dataset = None, n_clusters = None, distance_threshold = 10.0, compute_full_tree = True, metric = "euclidean", linkage = "ward", save_model = True):
        """
        Train AgglomerativeClustering model

        :param dataset: dataset for model training. Default to self.DATASET
        :param n_clusters: int, number of clusters
        :param distance_threshold: float, minimum distance between each cluster
        :param compute_full_tree: bool
        :param metric: str, distance metric for model
        :param linkage: str, linkage method
        :param save_model: bool, indicate whether to save the trained model or not
        :return: tuple (AgglomerativeClustering model, number of clusters, cluster label for each data)
        """
        # default to class property if train_set are not provided
        if not dataset:
            dataset = self.DATASET
        
        # train the model
        print(f"[INFO] Training AgglomerativeClustering model")
        model = AgglomerativeClustering(
                n_clusters = n_clusters,
                distance_threshold = distance_threshold if not n_clusters else None,
                compute_full_tree = compute_full_tree,
                metric = metric,
                linkage = linkage
        )
        cluster_label = model.fit_predict(dataset)
        print("[INFO] Done")
        
        # save the model to class property
        self.MODEL = model
        print("[INFO] You may find the trained model in self.MODEL")
        
        # save the model to disk
        if save_model:
            joblib.dump(model, "AgglomerativeClustering_model.joblib")
            print("[INFO] Model saved as ./AgglomerativeClustering_model.joblib")
        
        return model, model.n_clusters_, cluster_label
    

class TSKMeans(Clustering):
    
    def __init__(self):
        """
        A class for time series K-means clustering

        Before working on the model,
        user should call either dataset() or train_test_validation_set() or dataset_with_metadata()
        to provide data for model training and evaluation using class methods
        """
        # Clustering.__init__(self)
        super().__init__()
    
    
    def find_optimal_k(self, k_range, train_set = None, validation_set = None, metric = "euclidean", random_state = 42):
        """
        Find optimal number of cluster k
        (hyperparameter tuning)

        :param k_range: tuple of int e.g. (3, 10), range of k to test
        :param train_set: train dataset for model training. Default to self.TRAIN_SET
        :param validation_set: validation set. Default to self.VALIDATION_SET
        :param metric: str, distance metric for model e.g. euclidean, correlation, manhattan
        :param random_state: int
        :return: tuple (int: optimal k, float: silhouette score)
        """
        # default to class property if train_set and validation_set are not provided
        if not train_set:
            train_set = self.TRAIN_SET
            # train_set = TimeSeriesScalerMeanVariance().fit_transform(train_set)
        
        if not validation_set:
            validation_set = self.VALIDATION_SET
            # validation_set = TimeSeriesScalerMeanVariance().fit_transform(validation_set)
        
        # find optimal number of cluster k
        optimal_silhouette_k = 2
        optimal_dbi_k = 2
        silhouette_score_ = -1  # silhouette score find max, greatest separation between clusters
        davies_bouldin_score_ = 1e3  # find min
        inertia = []  # Elbow Method
        
        print("[INFO] Finding optimal value of k")
        print(f"k  |  Silhouette Score  |  Davies Bouldin (DBI) Score  |  Inertia")
        
        for k in range(k_range[0], k_range[1] + 1):
            np.random.seed(random_state)
            _model = TimeSeriesKMeans(
                    n_clusters = k,
                    metric = metric,
                    # method = "alternate",
                    random_state = random_state,
                    max_iter = 10
            )
            _model.fit(train_set)  # train model on train_set
            _label = _model.predict(validation_set)  # predict data on validation_set
            
            # evaluate how well the validation data fit into trained clusters
            _silhouette_score = silhouette_score(validation_set, _label, metric = metric)
            _davies_bouldin_score = davies_bouldin_score(validation_set, _label)
            print(f"{k} | {_silhouette_score} | {_davies_bouldin_score} | {_model.inertia_}")
            
            if _silhouette_score > silhouette_score_:
                silhouette_score_ = _silhouette_score
                optimal_silhouette_k = k
            
            if _davies_bouldin_score < davies_bouldin_score_:
                davies_bouldin_score_ = _davies_bouldin_score
                optimal_dbi_k = k
            
            # else:
            #     print(f"Found optimal k = {optimal_k}")
            #     break
        
        print(f"Optimal k for Silhouette Score : {optimal_silhouette_k}")
        print(f"Optimal k for Davies Bouldin (DBI) Score : {optimal_dbi_k}")
        
        return optimal_silhouette_k, silhouette_score_  # todo
    
    
    def fit(self, k, train_set = None, metric = "euclidean", random_state = 42, save_model = True):
        """
        Train K-Medoids model

        :param k: int, number of clusters
        :param train_set: train dataset for model training. Default to self.TRAIN_SET
        :param metric: str, distance metric for model
        :param random_state: int
        :param save_model: bool, indicate whether to save the trained model or not
        :return: the trained K-Medoids model
        """
        # default to class property if train_set are not provided
        if not train_set:
            # train_set = self.TRAIN_SET
            train_set = np.vstack((self.TRAIN_SET, self.VALIDATION_SET))
            # scale / normalize the data
            # train_set = TimeSeriesScalerMeanVariance().fit_transform(train_set)
        
        # train the model
        print(f"[INFO] Training Time Series K-Means model with k = {k} clusters")
        model = TimeSeriesKMeans(
                n_clusters = k,
                metric = metric,
                random_state = random_state
        )
        model.fit(train_set)
        print("[INFO] Done")
        
        # save the model to class property
        self.MODEL = model
        print("[INFO] You may find the trained model in self.MODEL")
        
        # save the model to disk
        if save_model:
            joblib.dump(model, "K-Medoids_model.pkl")
            print("[INFO] Model saved as ./K-Medoids_model.pkl")
        
        return model
    
    
    def predict(self, model = None, test_set = None, metric = "euclidean"):
        """
        Model evaluation on unseen test set ( after fit() )
        with fixed cluster center

        :param model: the K-Medoids model. Default to self.MODEL
        :param test_set: test dataset for model evaluation. Default to self.TEST_SET
        :param metric: str, distance metric for model
        :return: tuple (cluster labels, silhouette score)
        """
        # default to class property if model and test_set are not provided
        if not model:
            model = self.MODEL
        
        if test_set is None:
            test_set = self.TEST_SET
            # test_set = TimeSeriesScalerMeanVariance().fit_transform(test_set)
        
        test_labels = model.predict(test_set)
        silhouette_score_ = silhouette_score(test_set, test_labels, metric = metric)
        print(f"[INFO] Test Set Silhouette Score: {silhouette_score_:.4f}")
        
        return test_labels, silhouette_score_
    

class MiniBatchKMean(Clustering):
    
    def __init__(self):
        """
        A class for Mini Batch K-means clustering

        Before working on the model,
        user should call either dataset() or train_test_validation_set() or dataset_with_metadata()
        to provide data for model training and evaluation using class methods
        """
        # Clustering.__init__(self)
        super().__init__()
        self.clusters = None
    
    
    def find_optimal_k(self, k_range, train_set = None, validation_set = None, batch_size = 256, random_state = 42):
        """
        Find optimal number of cluster k
        (hyperparameter tuning)

        :param k_range: tuple of int e.g. (3, 10), range of k to test
        :param train_set: train dataset for model training. Default to self.TRAIN_SET
        :param validation_set: validation set. Default to self.VALIDATION_SET
        :param batch_size: int
        :param random_state: int
        :return: tuple (int: optimal k, float: silhouette score)
        """
        # default to class property if train_set and validation_set are not provided
        if not train_set:
            train_set = self.TRAIN_SET
        
        if not validation_set:
            validation_set = self.VALIDATION_SET
        
        # find optimal number of cluster k
        optimal_silhouette_k = 2
        optimal_dbi_k = 2
        silhouette_score_ = -1  # silhouette score find max, greatest separation between clusters
        inertia = []  # Elbow Method
        davies_bouldin_score_ = 1e3  # find min
        
        print("[INFO] Finding optimal value of k")
        print(f"k  |  Silhouette Score  |  Davies Bouldin (DBI) Score  |  Inertia")
        
        for k in range(k_range[0], k_range[1] + 1):
            np.random.seed(random_state)
            _model = MiniBatchKMeans(
                    n_clusters = k,
                    batch_size = batch_size,
                    random_state = random_state,
                    n_init = "auto"
            )
            _model.fit(train_set)  # train model on train_set
            _label = _model.predict(validation_set)  # predict data on validation_set
            
            # evaluate how well the validation data fit into trained clusters
            _silhouette_score = silhouette_score(validation_set, _label)
            _davies_bouldin_score = davies_bouldin_score(validation_set, _label)
            print(f"{k} | {_silhouette_score} | {_davies_bouldin_score} | {_model.inertia_}")
            
            if _silhouette_score > silhouette_score_:
                silhouette_score_ = _silhouette_score
                optimal_silhouette_k = k
            
            if _davies_bouldin_score < davies_bouldin_score_:
                davies_bouldin_score_ = _davies_bouldin_score
                optimal_dbi_k = k
            
            # else:
            #     print(f"Found optimal k = {optimal_k}")
            #     break
        
        print(f"Optimal k for Silhouette Score : {optimal_silhouette_k}")
        print(f"Optimal k for Davies Bouldin (DBI) Score : {optimal_dbi_k}")
        
        return optimal_silhouette_k, silhouette_score_  # todo
    
    
    def fit(self, k, train_set = None, batch_size = 256, random_state = 42, save_model = True):
        """
        Train Mini-batch K-Means model

        :param k: int, number of clusters
        :param train_set: train dataset for model training. Default to self.TRAIN_SET
        :param batch_size: int
        :param random_state: int
        :param save_model: bool, indicate whether to save the trained model or not
        :return: the trained Mini-batch K-Means model
        """
        # default to class property if train_set are not provided
        if not train_set:
            # train_set = self.TRAIN_SET
            train_set = np.vstack((self.TRAIN_SET, self.VALIDATION_SET))
        
        # train the model
        print(f"[INFO] Partial fitting Mini-batch K-Means model with k = {k} clusters")
        model = MiniBatchKMeans(
                n_clusters = k,
                batch_size = batch_size,
                random_state = random_state
        )
        model.partial_fit(train_set)
        print("[INFO] Done")
        
        # save the model to class property
        self.MODEL = model
        print("[INFO] You may find the trained model in self.MODEL")
        
        # save the model to disk
        if save_model:
            joblib.dump(model, "Mini-batch_K-Means_model.pkl")
            print("[INFO] Model saved as ./Mini-batch_K-Means_model.pkl")
        
        return model
    
    
    def predict(self, model = None, test_set = None, batch_size = 256):
        """
        Model evaluation on unseen test set ( after fit() )
        with fixed cluster center

        :param model: the Mini-batch K-Means model. Default to self.MODEL
        :param test_set: test dataset for model evaluation. Default to self.TEST_SET
        :param metric: str, distance metric for model
        :return: tuple (cluster labels, silhouette score)
        """
        # default to class property if model and test_set are not provided
        if not model:
            model = self.MODEL
        
        if test_set is None:
            test_set = self.TEST_SET
        
        predicts = []
        batch = []
        
        for data in test_set:
            predicts.append(np.argmin(np.linalg.norm(self.clusters - data, axis = 1)))
            batch.append(data)
            
            if len(batch) >= batch_size:
                model.partial_fit(np.array(batch))
                self.clusters = model.cluster_centers_
                batch = []
                
        # test_labels = model.predict(test_set)
        # silhouette_score_ = silhouette_score(test_set, test_labels)
        # print(f"[INFO] Test Set Silhouette Score: {silhouette_score_:.4f}")
        
        # return test_labels, silhouette_score_
        return predicts


class PypotsCRLI(Clustering):
    
    def __init__(self):
        """
        A class for CRLI (from PyPots) clustering
    
        Before working on the model,
        user should call either dataset() or train_test_validation_set() or dataset_with_metadata()
        to provide data for model training and evaluation using class methods
        """
        # Clustering.__init__(self)
        super().__init__()
        self.n_samples = None  # self.DATASET.shape[0]  # number of data points (rows), assumed 2D dataset
        self.n_steps = None  # self.DATASET.shape[1]  # number of data in each data points (columns)
        self.n_features = 1
        
        self.n_clusters = None
        
        self.batch_size = 256
        self.epoch = 30
        self.n_generator_layers = 2
        self.rnn_hidden_size = 128
        self.rnn_cell_type = "LSTM"  # GRU / LSTM
        self.device = "cpu"
        
        
    def find_optimal_k(self, k_range, train_set = None, validation_set = None, **kwargs):
        # default to class property if train_set and validation_set are not provided
        if not train_set:
            train_set = self.TRAIN_SET
        
        if not validation_set:
            validation_set = self.VALIDATION_SET
            
        # find optimal number of cluster k
        chs_score = 1e-3  # find max
        dbs_score = 1e3  # find min
        optimal_k_chs = 0
        optimal_k_dbs = 0
        optimal_k = 2
        
        print("[INFO] Finding optimal value of k")
        print(f"k  |  Silhouette Score  |  Davies Bouldin (DBI) Score  |  Inertia")
        
        for k in range(10, 30):
            print(f"Trying k = {k}")
            model = CRLI(
                    n_steps = self.n_steps,
                    n_features = self.n_features,
                    n_clusters = k,
                    epochs = self.epoch,
                    n_generator_layers = self.n_generator_layers,
                    rnn_hidden_size = self.rnn_hidden_size,
                    rnn_cell_type = self.rnn_cell_type,
                    batch_size = self.batch_size,  # lower batch size to prevent memory overloading for large dataset
                    # patience = 5,  # Early stopping patience
                    device = self.device,
                    **kwargs
            )
            model.fit(train_set = train_set)
            predictions = model.predict(validation_set)
            labels = predictions["clustering"]
            imputed_features = predictions["imputed_X"].reshape(self.n_samples, -1)
            _chs_score = calc_chs(imputed_features, labels)
            _dbs_score = calc_dbs(imputed_features, labels)
            
            print(f"{k}  | ")
            
            if _chs_score > chs_score:
                chs_score = _chs_score
                optimal_k_chs = k
            
            if _dbs_score < dbs_score:
                dbs_score = _dbs_score
                optimal_k_dbs = k
            
            if k > max(optimal_k_chs, optimal_k_dbs):
                optimal_k = k - 1
                
        self.n_clusters = optimal_k
        print(f"[INFO] Optimal number of clusters {optimal_k} stored to class property")
        
        return optimal_k
    
    
    def fit(self, train_set = None, save_model = True, **kwargs):
        """
        Train CRLI model
        
        :param train_set: train dataset for model training. Default to self.TRAIN_SET
        :param save_model: bool, indicate whether to save the trained model or not
        :return: the trained CRLI model
        """
        # default to class property if train_set are not provided
        if not train_set:
            # train_set = self.TRAIN_SET
            train_set = np.vstack((self.TRAIN_SET, self.VALIDATION_SET))
            
        # train the model
        print(f"[INFO] Training CRLI model with k = {self.n_clusters} clusters")
        model = CRLI(
                n_steps = self.n_steps,
                n_features = self.n_features,
                n_clusters = self.n_clusters,
                epochs = self.epoch,
                n_generator_layers = self.n_generator_layers,
                rnn_hidden_size = self.rnn_hidden_size,
                rnn_cell_type = self.rnn_cell_type,
                batch_size = self.batch_size,  # lower batch size to prevent memory overloading for large dataset
                # patience = 5,  # Early stopping patience
                device = self.device,
                **kwargs
        )
        model.fit(train_set)
        print("[INFO] Done")
        
        # save the model to class property
        self.MODEL = model
        print("[INFO] You may find the trained model in self.MODEL")
        
        # save the model to disk
        if save_model:
            # joblib.dump(model, "CRLI_model.pypots")
            model.save("CRLI_model.pypots")
            print("[INFO] Model saved as ./CRLI_model.pypots")
        
        return model
    
    
    def auto_partial_fit(self, dataset = None, save_model = True, **kwargs):
        # default to class property if dataset are not provided
        if not dataset.any():
            dataset = self.DATASET
            
        temp_file_name = "temp_dataset.h5"
        
        if not os.path.exists(temp_file_name):
            
            with h5py.File(temp_file_name, "w") as file:
                print(f"[INFO] Created temporary file {temp_file_name}")
                X = file.create_dataset("X", shape = (self.n_samples, int(self.n_steps / 100), self.n_features), dtype = "float32")
                print("[INFO] Writing data to file")
                
                for i in range(self.n_samples):
                    _night_data = dataset[i]
                    # _compressed_night_data = _night_data.reshape(int(self.n_steps / 100), 100).mean(axis = 1)
                    _compressed_night_data = signal.resample(_night_data, int(self.n_steps / 100))
                    X[i, :, 0] = _compressed_night_data
                
                file.close()
        
        model = self.fit(train_set = temp_file_name, save_model = save_model, **kwargs)
        
        return model
    
    
    def predict(self):
        pass
    
    
"""
NSB = NightSkyBrightness(
        "iyl/sqmle_nsb_data_table/HKU.csv",
        "src/Sun/Sun_rise_set_2018.csv",
        "src/Moon/Moon_rise_set_2018.csv"
)
print(NSB.get_sun_set_time("2018-01-02"))
print(NSB.get_sun_rise_time("2018-01-02"))
print(NSB.get_moon_set_time("2018-01-02"))
print(NSB.get_moon_rise_time("2018-01-02"))
print(NSB.get_moon_transition_time("2018-01-02"))
print(NSB.get_moon_phase("2018-01-02"))
print(NSB.get_moon_phase("2026-01-19 03:52:00"))
# NSB.get_one_night_data("2018-01-01 03:52:20")
# print(NSB.one_night_average("2018-12-31"))
# print(NSB.one_month_average("2018-01-02"))
# print(NSB.one_year_average("2018-02-02"))  # error
"""
'''
import math

def julian_date(year, month, day):
    """Convert a calendar date to Julian Date."""
    if month <= 2:
        year -= 1
        month += 12
    A = math.floor(year / 100)
    B = 2 - A + math.floor(A / 4)
    JD = math.floor(365.25 * (year + 4716)) \
         + math.floor(30.6001 * (month + 1)) \
         + day + B - 1524.5
    return JD

def moon_phase_duffett(year, month, day):
    """
    Calculate moon phase using Duffett-Smith's 1981 algorithm.
    Returns:
        phase_name (str), illumination_percent (float), phase_fraction (float)
    """
    # Julian date for given date
    jd = julian_date(year, month, day)

    # Days since epoch (1900 Jan 0.5)
    days_since_epoch = jd - 2415020.0

    # Mean anomaly of the Sun
    N = (360 / 365.2422) * days_since_epoch
    N = N % 360

    # Mean anomaly of the Moon
    M_sun = N
    M_moon = (360 / 27.321582) * days_since_epoch
    M_moon = M_moon % 360

    # Sun's mean longitude
    L_sun = M_sun + 279.696678
    L_sun = L_sun % 360

    # Moon's mean longitude
    L_moon = M_moon + 270.434164
    L_moon = L_moon % 360

    # Moon's mean anomaly
    M_moon_anomaly = L_moon - 0.111404 * days_since_epoch - 318.15
    M_moon_anomaly = M_moon_anomaly % 360

    # Moon's argument of latitude
    F = L_moon - L_sun
    F = F % 360

    # Perturbations
    # Convert to radians for trig functions
    M_sun_rad = math.radians(M_sun)
    M_moon_anomaly_rad = math.radians(M_moon_anomaly)
    F_rad = math.radians(F)

    # Correction terms from Duffett-Smith
    corr = -0.1734 * math.sin(M_moon_anomaly_rad) \
           - 0.0021 * math.sin(2 * M_moon_anomaly_rad) \
           + 0.0161 * math.sin(M_sun_rad) \
           + 0.0004 * math.sin(2 * M_sun_rad) \
           - 0.0005 * math.sin(F_rad)

    # Phase angle (degrees)
    phase_angle = F + corr
    phase_angle = phase_angle % 360

    # Fraction of cycle (0 = New Moon, 0.5 = Full Moon)
    phase_fraction = (1 - math.cos(math.radians(phase_angle))) / 2

    # Illumination percentage
    illumination_percent = phase_fraction * 100

    # Phase name mapping
    phase_names = [
        "New Moon",
        "Waxing Crescent",
        "First Quarter",
        "Waxing Gibbous",
        "Full Moon",
        "Waning Gibbous",
        "Last Quarter",
        "Waning Crescent"
    ]
    phase_index = int(phase_fraction * 8 + 0.5) % 8

    return phase_names[phase_index], illumination_percent, phase_fraction

# Epoch Reference:
# Uses 1900 Jan 0.5 as the base Julian date (2415020.0).
# Mean Anomalies:
# Calculates Sun and Moon mean anomalies separately.
# Perturbations:
# Applies small corrections for orbital eccentricity and latitude.
# Phase Angle:
# Determines the angular separation between Sun and Moon.
# Illumination:
# Uses cosine of phase angle to get illuminated fraction.
'''


'''
# TS3C (trying)
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import skew


# =====================================================================
# STAGE 1: 滾動視窗多項式分割與特徵提取 (尊重夜空光度原始微觀特徵)
# =====================================================================
def swiftseg_polynomial_fit(time_series, max_error = 0.1, degree = 1):
    n = len(time_series)
    segments = []
    start = 0
    
    while start < n - 2:
        end = start + 2
        
        while end < n:
            x = np.arange(start, end + 1)
            y = time_series[start:end + 1]
            coeffs = np.polyfit(x, y, degree)
            y_pred = np.polyval(coeffs, x)
            
            error = np.max(np.abs(y - y_pred))
            
            if error > max_error:
                break
                
            end += 1
        
        final_x = np.arange(start, end)
        final_y = time_series[start:end]
        final_coeffs = np.polyfit(final_x, final_y, degree)
        
        # 提取光度統計特徵：[斜率, 截距, 光度方差(雲層波動), 偏度, 一階自相關]
        variance = np.var(final_y)
        if variance < 1e-6: variance = 1e-6
        segment_skew = skew(final_y) if len(final_y) > 2 else 0.0
        if np.isnan(segment_skew) or np.isinf(segment_skew): segment_skew = 0.0
        autocorr = np.corrcoef(final_y[:-1], final_y[1:]) if len(final_y) > 2 else 0.0
        
        if np.isnan(autocorr).any() or isinstance(autocorr, float):
            autocorr = 0.0
        else:
            autocorr = autocorr[0, 1]
        
        feature_vector = np.concatenate([final_coeffs, [variance, segment_skew, autocorr]])
        segments.append(feature_vector)
        start = end
    
    return np.array(segments)


#
# Find optimal number of cluster
#
def find_optimal_clusters(X_dataset, max_error = 0.2, max_k_seg = 15, max_k_final = 5):
    """
    自動計算並輸出第一階段與第二階段的最佳分群數
    """
    # --- STAGE 1: 提取所有區段並攤平 ---
    all_segments_list = []
    series_to_seg_mapping = []
    
    for ts in X_dataset:
        seg_features = swiftseg_polynomial_fit(ts, max_error = max_error)
        all_segments_list.append(seg_features)
        series_to_seg_mapping.append(len(seg_features))
        
    flat_segments = np.vstack(all_segments_list)
    
    # --- 優化第一階段：Segment Clustering (使用 Calinski-Harabasz 指數) ---
    print("🔍 正在評估第一階段（局部夜空行為）的最佳分群數...")
    best_k_seg = 2
    best_ch_score = -1
    
    # 預先計算層次分群樹（Ward 連結），避免重複計算，大幅降低運算成本
    seg_linkage_matrix = linkage(flat_segments, method = 'ward')
    
    for k in range(2, min(max_k_seg + 1, len(flat_segments))):
        print(f"Trying k_segment = {k}")
        # 直接從樹狀矩陣中切出 K 個分群，速度極快
        labels = fcluster(seg_linkage_matrix, t = k, criterion = 'maxclust')
        score = calinski_harabasz_score(flat_segments, labels)
        
        if score > best_ch_score:
            best_ch_score = score
            best_k_seg = k
            
    print(f"🌟 建議最佳【局部區段分群數 (k_segment)】: {best_k_seg} (CH Score: {best_ch_score:.2f})")
    
    # 使用最佳的 k_seg 標籤來建立第二階段的 Profile
    optimal_seg_labels = fcluster(seg_linkage_matrix, t = best_k_seg, criterion = 'maxclust') - 1
    
    # --- STAGE 2: 建立每條序列的特徵 Profile ---
    mapped_series_features = []
    current_idx = 0
    
    for num_segs in series_to_seg_mapping:
        ts_seg_labels = optimal_seg_labels[current_idx: current_idx + num_segs]
        current_idx += num_segs
        hist, _ = np.histogram(ts_seg_labels, bins = np.arange(best_k_seg + 1))
        hist_ratio = hist / num_segs if num_segs > 0 else np.zeros(best_k_seg)
        mapped_series_features.append(hist_ratio)
        
    mapped_series_features = np.array(mapped_series_features)
    
    # --- 優化第二階段：Final Clustering (使用 Silhouette Score) ---
    print("\n🔍 正在評估第二階段（觀測站/全序列）的最佳分群數...")
    best_k_final = 2
    best_silhouette = -1
    
    # 限制最大測試 K 值不能超過總序列數 - 1
    actual_max_k_final = min(max_k_final, len(X_dataset) - 1)
    
    if actual_max_k_final >= 2:
        final_linkage_matrix = linkage(mapped_series_features, method = 'ward')
        
        for k in range(2, actual_max_k_final + 1):
            print(f"Trying k = {k}")
            labels = fcluster(final_linkage_matrix, t = k, criterion = 'maxclust')
            score = silhouette_score(mapped_series_features, labels)
            
            if score > best_silhouette:
                best_silhouette = score
                best_k_final = k
                
            else:
                break
                
        print(f"🌟 建議最佳【最終全序列分群數 (k_final)】: {best_k_final} (Silhouette Score: {best_silhouette:.2f})")
    
    else:
        print("⚠️ 序列樣本數過少，默認最終分群數為 2")
        best_k_final = 2
    
    return best_k_seg, best_k_final

# =====================================================================
# TS3C 管線：全層次分群架構 (Hierarchical-based TS3C)
# =====================================================================
class TS3C_Hierarchical_Clustering:
    
    def __init__(self, num_segment_clusters = 5, num_final_clusters = 3, max_error = 0.15):
        self.num_segment_clusters = num_segment_clusters  # 局部光度行為型態數（如：受月光干擾段、無雲暗空段）
        self.num_final_clusters = num_final_clusters  # 最終夜空序列或站點的分群數
        self.max_error = max_error
        
        # 第一階段：對局部區段進行層次分群 (使用 ward 連接法，極適合連續物理特徵)
        self.segment_clustering = KMeans(n_clusters=num_segment_clusters, random_state=42)
        # 第二階段：對全序列行為 Profile 進行層次分群
        self.final_clustering = AgglomerativeClustering(n_clusters = num_final_clusters, linkage = 'ward')
    
    def fit_predict(self, X_dataset):
        all_segments_list = []
        series_to_seg_mapping = []
        
        print("🌙 [Stage 1] 正在提取夜空光度原始區段特徵...")
        for ts in X_dataset:
            seg_features = swiftseg_polynomial_fit(ts, max_error = self.max_error)
            all_segments_list.append(seg_features)
            series_to_seg_mapping.append(len(seg_features))
        
        flat_segments = np.vstack(all_segments_list)
        print(f" -> 原始數據轉換完成。共生成 {flat_segments.shape[0]} 個局部光度特徵區段。")
        
        # 執行第一階段層次分群
        print("⚡ [Stage 1] 正在對局部夜空行為進行層次分類 (Agglomerative)...")
        segment_labels = self.segment_clustering.fit_predict(flat_segments)
        
        # --- STAGE 2: 重構每條夜空序列的行為分佈 ---
        print("🔮 [Stage 2] 正在重構觀測站/序列的全局特徵 Profile...")
        mapped_series_features = []
        current_idx = 0
        
        for num_segs in series_to_seg_mapping:
            ts_seg_labels = segment_labels[current_idx: current_idx + num_segs]
            current_idx += num_segs
            
            # 計算每條夜空序列在各類光度行為上的機率直方圖 (代表該站點的夜空氣候特徵)
            hist, _ = np.histogram(ts_seg_labels, bins = np.arange(self.num_segment_clusters + 1))
            hist_ratio = hist / num_segs if num_segs > 0 else np.zeros(self.num_segment_clusters)
            mapped_series_features.append(hist_ratio)
        
        mapped_series_features = np.array(mapped_series_features)
        
        # 執行最終的時間序列主體層次分群
        print("🏆 [Stage 2] 正在執行最終夜空站點層次群聚...")
        final_labels = self.final_clustering.fit_predict(mapped_series_features)
        
        return final_labels, mapped_series_features


# =====================================================================
# 測試運行 (模擬 3 種不同環境的夜空光度觀測站)
# =====================================================================
if __name__ == "__main__":
    dataset = []
    date = []
    time = []
    moon_illuminated_ratio = []
    moon_phase = []
    data_drop_count = 0
    data_length_tolerance = 2 * 60 * 2  # min number of data points per night,  # 30s 1 data point, at least 2 hours of data
    sun_set_time = "19:00:00"
    sun_rise_time = "05:30:00"
    
    db_class = DataBase("src/HKU/HKU.csv")
    db_HKU = db_class.read().drop_duplicates()
    
    # Pivot Data: Reshape into a nightly curve matrix
    # [       time1 time2 ...
    #   date1 nsb11 nsb12 ...
    #   date2 nsb21 nsb22 ... ]
    db_HKU_matrix = db_HKU.pivot(
            index = "astronomical_date",
            columns = "time",
            values = "nsb"
    )
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
    
    # drop column according to fixed sun set/rise time
    _filter = (
            (db_HKU_matrix.columns >= sun_set_time) |
            (db_HKU_matrix.columns <= sun_rise_time)
    )
    db_HKU_matrix = db_HKU_matrix.loc[:, _filter]
    
    # save the time
    time = db_HKU_matrix.columns
    
    print("[INFO] Collecting data")
    
    for _date, _group in db_HKU.groupby("astronomical_date"):
        _data = np.copy(db_HKU_matrix.loc[_date].values)  # numpy array
        
        if len(_data[~np.isnan(_data)]) <= data_length_tolerance:
            # drop night with insufficient data points
            data_drop_count += 1
            continue
        
        # linear interpolate data
        # https://stackoverflow.com/questions/38064697/interpolating-a-numpy-array-to-fit-another-array
        # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
        _data[np.isnan(_data)] = np.interp(
                np.isnan(_data).nonzero()[0],
                (~np.isnan(_data)).nonzero()[0],
                _data[~np.isnan(_data)]
        )
        
        # append to data structure
        dataset.append(_data)
        date.append(_date)
        moon_illuminated_ratio.append(_group["moon_illuminated_ratio"])
        moon_phase.append(_group["moon_phase"])
    
    print(f"[INFO] Dropped {data_drop_count} night(s) of data")
    
    
    
    # 執行全層次 TS3C 分群
    mock_nsb_data = np.array(dataset)
    index = np.random.choice(len(mock_nsb_data), 365, replace = False)
    train_data = []
    for i in index:
        train_data.append(mock_nsb_data[i])
    
    opt_seg, opt_final = find_optimal_clusters(
            train_data,
            max_error = 0.15,
            max_k_seg = 20,
            max_k_final = 30
    )
    ts3c_hr = TS3C_Hierarchical_Clustering(
            num_segment_clusters = opt_seg,
            num_final_clusters = opt_final,
            max_error = 0.2
    )
    labels, profiles = ts3c_hr.fit_predict(mock_nsb_data)
    
    print("\n================ 最終夜空分群結果 ================")
    cluster_data = {}
    for i in range(opt_final):
        cluster_data[i] = {"data": [], "date": []}
        
    for i, label in enumerate(labels):
        cluster_data[i]["data"].append(mock_nsb_data[i])
        cluster_data[i]["date"].append(date[i])
        print(
            f"📡 Day #{i + 1} (data length: {len(mock_nsb_data[i])}) -> 分類至: {label} (真實夜空特徵已保留)")
'''

