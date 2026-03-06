import pymysql

def get_db_connection():
    connection = pymysql.connect(
        host="localhost",
        user="root",
        password="",  # put your MySQL password here if you set one
        database="customeranalysis",
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection