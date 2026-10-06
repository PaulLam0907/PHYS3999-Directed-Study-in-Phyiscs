import datetime

from database import DataBase
from figure import Figure
from nsb import NightSkyBrightness


def plot_one_month_nsb(fig, month_average, daily_average, moon_phase, title, axes_index = 0):
    fig.plot([i for i in range(1, len(daily_average) + 1)], daily_average, axes_index, label = "NSB")
    fig.plot_horizontal_line(month_average, axes_index, label = "Month average NSB")
    
    moon_phase = enumerate(moon_phase, start = 1)
    phase = ""
    
    for i in moon_phase:
        
        if phase != i[1][0]:
            fig.plot_vertical_line(i[0], axes_index, label = i[1][0])
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
            legend = True
    )


# Plot 2025 January and June average
NSB_HKU_2025 = NightSkyBrightness(
        "iyl/sqmle_nsb_data_table/HKU.csv",
        "src/Sun/Sun_rise_set_2025.csv",
        "src/Moon/Moon_rise_set_2025.csv"
)

# january_data = NSB_HKU_2025.one_month_average("2025-01-01")
# june_data = NSB_HKU_2025.one_month_average("2025-06-01")

# fig = Figure("2025 NSB", 1, 2)

# JANUARY
# plot_one_month_nsb(fig, january_data[0], january_data[1], january_data[2], "January", 0)
# JUNE
# plot_one_month_nsb(fig, june_data[0], june_data[1], june_data[2], "June", 1)

# fig.show()


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
