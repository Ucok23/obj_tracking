from datetime import datetime
from mysql.connector import Error, errorcode
from yolo.configs import CLASSES_TO_DETECT

# Database Config
DB_USER = 'root'
DB_PASSWORD = 'root'
DB_HOST = '127.0.0.1'
DB_NAME = 'track'


def create_database(cursor, db_name):
    try:
        cursor.execute(
            f"CREATE DATABASE {db_name} DEFAULT CHARACTER SET 'utf8'")
    except Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)


def setup_database(cursor, cx):
    try:
        cursor.execute(f"USE {DB_NAME}")
    except Error as err:
        print(f"Database {DB_NAME} does not exists.")
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            create_database(cursor, DB_NAME)
            print(f"Database {DB_NAME} created successfully.")
            cx.database = DB_NAME
            cursor.execute(f"USE {DB_NAME}")
        else:
            print(err)
            exit(1)


def setup_table(cursor, cx):
    setup_database(cursor, cx)

    table_desc_list = []
    for cls in CLASSES_TO_DETECT:
        desc = (
            f"CREATE TABLE `{cls}` ("
            "  `table_id` int(11) NOT NULL AUTO_INCREMENT,"
            "  `id` int(11) NOT NULL,"
            "  `time` datetime NOT NULL,"
            "  PRIMARY KEY (`table_id`)"
            ") ENGINE=InnoDB")
        table_desc_list.append(desc)

    try:
        for table in table_desc_list:
            cursor.execute(table)
    except Error as err:
        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("[INFO] Table already exists.")
        else:
            print(err.msg)


def insert_result(cursor, cx, table, data):
    now = datetime.now()
    query = (f"INSERT INTO {table} "
             "(id, time) "
             "VALUES (%(id)s, %(time)s)")
    print(query)
    data['time'] = now
    cursor.execute(query, data)
    cx.commit()
