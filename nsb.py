"""
A module for night-sky brightness (NSB) analysis

1. Slice data by time windows
2. K-Medoids (PAM) / K-Shape
3. map each night to cluster class
• The Yearly Composition Shift: Count how many nights belonged to each cluster for Year 1 vs. Year 10. If Year 1 had 60% "Deep Dark Nights" (Cluster 1) and Year 10 only has 20%, you have successfully isolated a clear, quantifiable light pollution trend.
• The Transition Matrix: Calculate the probability of moving from one cluster type to another over the years, showing if the night sky is degrading or stabilizing.
4. distance metrics: DTW distance / cross-correlation (Shape-Based Distance (SBD))

Package Requirement :
    numpy
    pandas
    tslearn
"""
import os
import datetime
import numpy as np
import pandas as pd
from tslearn import clustering  # .KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from database import DataBase
from figure import Figure


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
    
    
    def get_one_night_data(self, date_night_from):
        """
        Prepare one night data of NSB with Sun set/rise time adjusted
        Retrieve data from [the night of date_night_from] to [the dawn of next day of date_night_from]
        
        :param date_night_from: str, YYYY-MM-DD, accept YYYY-MM-DD HH:MM:SS
        :return: tuple (datetime, NSB, pandas.DataFrame)
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
                ["received_adjusted", "nsb"],
                filters = [
                    {"received_adjusted": [">=", datetime_from]},
                    {"received_adjusted": ["<=", datetime_to]},
                    # {"nsb": ["<", 23]}
                ]
        )
        print("[INFO] Data loaded to memory")
        
        x = data["received_adjusted"]
        y = data["nsb"]
        
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
    

class KShape:
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


if os.path.exists("src/HKU_interpolated.csv"):
    db_HKU_interpolated = DataBase("src/HKU_interpolated.csv").read(index_col = "astronomical_date")
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
    print(f"Number of raw data in cluster {i}: {len(_data)}")
    
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
