"""
A module for database processing

Package Requirement :
    (mysql-conector-python)
    pandas
"""
import gc
import os
import pandas as pd
# import mysql.connector


class DataBase:

    def __init__(self, local_db = "data.csv"):
        """
        A class for database manipulation
        Upon declaration of this class, user should specify the file name of working database

        :param local_db: str, file name of local database (with extension), assumed csv
        """
        self.LOCAL_DB = local_db
        # self.DATA_FRAME = None  # self.read()


    def _download(self):
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


    def read(self, **kwargs):
        """
        Read local data to pandas dataframe

        :return: new copy of pandas.DataFrame
        """
        data_frame = pd.read_csv(self.LOCAL_DB, **kwargs)
        
        return data_frame


    def get_info(self):
        """
        Get info of database

        :return:
        """
        # load data frame if not exists
        if not hasattr(self, "DATA_FRAME"):
            self.DATA_FRAME = self.read()
            
        return self.DATA_FRAME.info


    def get_header(self):
        """
        Get headers' name of database

        :return: list of str
        """
        # list(self.DATA_FRAME.columns)
        return self.read(nrows = 0).columns.tolist()
    
    
    def get_column(self, column_name, filters = None):
        """
        Get columns' data from database

        :param column_name: list of str, name/header of column to select
        :param filters: list of dict [{column: [comparison operator, filter]}, ...] for filtering selected column
        :return: new copy of pandas.DataFrame
        """
        # load data frame if not exists
        # if not hasattr(self, "DATA_FRAME"):
        #     self.DATA_FRAME = self.read()
        
        # check if column exists
        for col in column_name:

            if col not in self.get_header():  # self.DATA_FRAME.columns:
                raise ValueError(f"[ERROR] Column {col} not found\n"
                                 f"Available columns : {self.get_header()}")
            
        # apply filter
        if filters:
            filter_column = list(set(key for filter_dict in filters for key in filter_dict.keys()))
            result = self.read(usecols = column_name + filter_column)  # self.DATA_FRAME.copy()
            _filter = pd.Series(True, index = result.index)
            
            for i in filters:
                
                for col, condition in i.items():
                    
                    if condition[0] not in ["=", "==", ">=", "<=", ">", "<", "!="]:
                        raise SyntaxError(f"Invalid Comparison Operator {condition[0]}")
                    
                    if condition[0] in ["=", "=="]:
                        # result = result[ result[col] == condition[1] ]
                        _filter &= (result[col] == condition[1])
                        
                    elif condition[0] == ">=":
                        # result = result[ result[col] >= condition[1] ]
                        _filter &= (result[col] >= condition[1])
                        
                    elif condition[0] == "<=":
                        # result = result[ result[col] <= condition[1] ]
                        _filter &= (result[col] <= condition[1])
                        
                    elif condition[0] == ">":
                        # result = result[ result[col] > condition[1] ]
                        _filter &= (result[col] > condition[1])
                        
                    elif condition[0] == "<":
                        # result = result[ result[col] < condition[1] ]
                        _filter &= (result[col] < condition[1])
                        
                    elif condition[0] == "!=":
                        # result = result[ result[col] != condition[1] ]
                        _filter &= (result[col] != condition[1])
            
            # return result[column_name]
            result = result.loc[_filter, column_name]
            
            if result.empty:
                print(f"[WARNING] Empty dataframe in getting columns {column_name} with filters {filters}")
                
            return result
        
        return self.read(usecols = column_name)  # self.DATA_FRAME[column_name]
    
    
    def close(self):
        """
        Clear computer memory
        
        :return:
        """
        
        if hasattr(self, "DATA_FRAME"):
            del self.DATA_FRAME
            
        gc.collect()
    
    
    def export(self, output_file_name = "output.csv", dataframe = None, column_name = None, filters = None, encoding = "utf-8"):
        """
        Export pandas.DataFrame or selected column(s) in self.LOCAL_DB
        
        :param output_file_name: str, file name of outputted CSV file (with extension)
        :param dataframe: pandas.DataFrame, the exact dataframe to export
        :param column_name: list of str, name/header of column to select
        :param filters: list of dict [{column: [comparison operator, filter]}, ...] for filtering selected column
        :param encoding: encoding of the outputted CSV file
        :return:
        """
        # validate parameters' value
        # if (not column_name) and (not filters):
        #     both dataframe, column, filters are not specified
        #     raise ValueError("[ERROR] Specify at least one parameter from \"dataframe\", \"column\" and \"filters\" to export as new dataset.")
        
        # export to CSV
        if isinstance(dataframe, pd.DataFrame):
            dataframe.to_csv(
                    output_file_name,
                    index = False,  # prevent writing row index number
                    encoding = encoding
            )
        
        elif column_name:
            # column is specified
            # filters is optional
            result = self.get_column(column_name, filters)
            result.to_csv(
                    output_file_name,
                    index = False,  # prevent writing row index number
                    encoding = encoding
            )
        
        elif filters:
            # only filters is provided
            column = self.get_header()
            result = self.get_column(column, filters)
            result.to_csv(
                    output_file_name,
                    index = False,  # prevent writing row index number
                    encoding = encoding
            )
            
        else:
            raise ValueError("[ERROR] Specify at least one parameter from \"dataframe\", \"column\" and \"filters\" to export as new dataset.")
        
        print(f"[INFO] Exported {output_file_name}")
        
        return True
        
    
    ########## v---- should be in separate class for better logic ##########
    def append(self, data_frame1, data_frame2, *args):
        column = data_frame1.columns.tolist()
        result = pd.merge(data_frame1, data_frame2, on = column, how = "outer")
        
        if args:
            for df in args:
                result = pd.merge(result, df, on = column, how = "outer")
        
        return result
    
    
    def concat(self, dataframe_list, ignore_index = True):
        return pd.concat(dataframe_list, ignore_index = ignore_index)
    
    
    def merge_csv(self, csv_file_name_list, output_file_name = "output.csv"):
        db_list = []
        
        for i in csv_file_name_list:
            self.LOCAL_DB = i
            db_list.append(self.read())
            
        result = pd.concat(db_list, ignore_index = True)
        result.to_csv(output_file_name, index = False)
        
        return True


"""
# merge csv
DataBase().merge_csv(
        [
            "src/Sun/Sun_rise_set_2018.csv",
            "src/Sun/Sun_rise_set_2019.csv",
            "src/Sun/Sun_rise_set_2020.csv",
            "src/Sun/Sun_rise_set_2020.csv",
            "src/Sun/Sun_rise_set_2021.csv",
            "src/Sun/Sun_rise_set_2022.csv",
            "src/Sun/Sun_rise_set_2023.csv",
            "src/Sun/Sun_rise_set_2024.csv",
            "src/Sun/Sun_rise_set_2025.csv",
            "src/Sun/Sun_rise_set_2026.csv",
        ],
        "src/Sun/Sun_rise_set_2018-2026.csv"
)
DataBase().merge_csv(
        [
            "src/Moon/Moon_rise_set_2018.csv",
            "src/Moon/Moon_rise_set_2019.csv",
            "src/Moon/Moon_rise_set_2020.csv",
            "src/Moon/Moon_rise_set_2020.csv",
            "src/Moon/Moon_rise_set_2021.csv",
            "src/Moon/Moon_rise_set_2022.csv",
            "src/Moon/Moon_rise_set_2023.csv",
            "src/Moon/Moon_rise_set_2024.csv",
            "src/Moon/Moon_rise_set_2025.csv",
            "src/Moon/Moon_rise_set_2026.csv",
        ],
        "src/Moon/Moon_rise_set_2018-2026.csv"
)
"""
