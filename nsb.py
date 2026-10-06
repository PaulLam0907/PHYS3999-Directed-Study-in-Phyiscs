"""
A module for night-sky brightness (NSB) analysis

Package Requirement :
    numpy
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
            "Last Quarter": [20.30, 23.99],
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
        :return: tuple (datetime, NSB)
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
                time_to = "00:00"  # fix December 31
        datetime_to = date_to + " " + time_to  # str
        
        # get data from db
        print("[INFO] Loading data from database")
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
        
        del data
        return x, y
    
    
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
