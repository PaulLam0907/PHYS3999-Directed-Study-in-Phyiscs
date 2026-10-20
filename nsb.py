"""
A module for night-sky brightness (NSB) analysis

1. Slice data by time windows
2. Algorithm :
    - K-Medoids (PAM) (force data into a cluster, cannot find outlier, uniform cluster density, require guessing k)
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
"""
import datetime
import numpy as np

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
                    # {"nsb": ["<", 23]}
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
# TS3C
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
    np.random.seed(42)
    # 模擬夜空光度數據 (不等長，數值代表 mag/arcsec²，數值愈大愈暗)
    mock_nsb_data = [
        # 站點 1 & 2：國際暗空公園 (高觀測品質，維持在 21.5 左右的高穩定暗度，受雲層影響微幅波動)
        21.5 + np.sin(np.linspace(0, 3, 500)) * 0.2 + np.random.normal(0, 0.05, 500),
        21.6 + np.sin(np.linspace(0, 3, 550)) * 0.15 + np.random.normal(0, 0.05, 550),
        
        # 站點 3 & 4：郊區觀測站 (中等光害，背景值在 19.5 左右，且因都市燈光反射，雲層一來光度會劇烈變亮/暴跌)
        19.5 - np.abs(np.random.normal(0, 0.8, 480)),
        19.3 - np.abs(np.random.normal(0, 0.7, 520)),
        
        # 站點 5 & 6：嚴重光害城市 (背景值極低約 17.0，且充滿人工光源的無規則隨機強光干擾)
        17.0 + np.random.normal(0, 0.4, 510),
        17.1 + np.random.normal(0, 0.35, 490)
    ]
    
    db_HKU = DataBase("src/HKU.csv").read()
    mock_nsb_data = []
    date = []
    data_drop = 0
    
    for _date, _group in db_HKU.groupby("astronomical_date"):
        _data = _group.iloc[:, 2].to_list()
        
        if len(_data) > 50 and not np.isnan(_data).any():
            mock_nsb_data.append(
                    _group.iloc[:, 2].to_list()
            )
            date.append(_date)
        
        else:
            data_drop += 1
            
    print(f"Dropped {data_drop} nights of data")
    
    # 執行全層次 TS3C 分群
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
