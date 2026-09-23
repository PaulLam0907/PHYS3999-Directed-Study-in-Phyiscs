import datetime

from database import DataBase
from figure import Figure


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
"""
plot_monthly_avg_2locations(  # memory inefficient & heavy
        db_ELO,
        db_PAOF,
        ["2020-01-01", "2021-01-01"],
        ["20:00:00", "05:00:00"],
        ["ELO", "PAOF"]
)
"""
# free up memory
db_ELO.close()
db_PAOF.close()
