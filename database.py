"""
Package Requirement :
    mysql-conector-python
    pandas
"""
import os
import pandas as pd
import mysql.connector


class DataBase:

    def __init__(self, local_db = "data.csv"):
        """
        A class for database manipulation
        Upon declaration of this class, user should specify the file name of working database

        :param local_db: str, file name of local database (with extension), assumed csv
        """
        self.LOCAL_DB = local_db
        self.DATA_FRAME = self.read()


    def download(self):
        """
        Download database from MySQL

        :return:
        """

        if not os.path.exists(self.LOCAL_DB):
            # no local copy of MySQL data exist
            # download from cloud
            print("[INFO] Login to MySQL to download local copy of data")

            # user login
            print("MySQL Login")
            USER = input("Username : ")
            PASSWORD = input("Password : ")
            DATABASE = input("Database (iyl) : ") or "iyl"

            try:
                # make connection to MySQL database
                connection = mysql.connector.connect(
                    host = "localhost",
                    port = 3306,
                    user = USER,
                    password = PASSWORD,
                    database = DATABASE
                )

                if connection.is_connected():
                    # successfully connected to MySQL database
                    print(f"Connected to MySQL Database {DATABASE}")
                    # cursor = connection.cursor()
                    # cursor.execute("")

                    # read data from MySQL to local memory
                    print(f"Reading data from MySQL Database {DATABASE}")
                    data_frame = pd.read_sql(f"SELECT * FROM {DATABASE}", connection)

                    # close connection
                    connection.close()
                    print("MySQL Connection Closed")

                    # store data to csv file locally
                    data_frame.to_csv(self.LOCAL_DB, index = False)


            except mysql.connector.Error as error:
                print(f"[ERROR] Failed to connect to MySQL Database {DATABASE}: \n{error}")


            finally:

                if "connection" in locals() and connection.is_connected():
                    # cursor.close()
                    connection.close()
                    print("MySQL Connection Closed")


        else:
            # local copy of MySQL data exist
            print(f"[INFO] Found local copy of data ({self.LOCAL_DB})")
            # data_frame = pd.read_csv(self.LOCAL_DB)


    def read(self):
        """
        Read local data to pandas dataframe

        :return: pandas.DataFrame
        """
        data_frame = pd.read_csv(self.LOCAL_DB)

        return data_frame


    def get_info(self):
        """
        Get info of database

        :return:
        """
        return self.DATA_FRAME.info


    def get_header(self):
        """
        Get headers' name of database

        :return:
        """
        # data_frame.columns

        return self.DATA_FRAME.head()


    def get_column(self, column_name, filters = dict):
        """
        Get columns' data from database

        :param column_name: list of str, name/header of column to select
        :param filters: dict {column: filter} for filtering selected column
        :return: new copy of pandas.DataFrame
        """
        # check if column exists
        for col in column_name:

            if col not in self.DATA_FRAME.columns:
                print(f"[ERROR] Column {col} not found")
                print(f"Available columns : {list(self.DATA_FRAME.columns)}")

                return None

        # apply filter
        if filters:
            result = self.DATA_FRAME.copy()

            for col, filter in filters.items():
                result = result[ result[col] == filter ]

            return result[column_name]

        return self.DATA_FRAME[column_name]

    