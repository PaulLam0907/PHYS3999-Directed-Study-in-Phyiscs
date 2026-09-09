from database import DataBase

db = DataBase("data.csv")
data_frame = db.read()
print(data_frame)
