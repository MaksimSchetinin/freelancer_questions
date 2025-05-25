import requests
import zipfile
import io
import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()


def load_dataset(zip_url: str, dataset_name: str, db_path: str):
    """
    Downloads a ZIP archive from a URL, extracts a CSV file from it,
    and stores the data into a SQLite database.

    :param zip_url: URL to the ZIP archive containing the dataset
    :param dataset_name: Name of the dataset (used for CSV file and DB table)
    :param db_path: Path to the SQLite database file
    """
    response = requests.get(zip_url)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        try:
            with zip_file.open(dataset_name + '.csv') as csv_file:
                with sqlite3.connect(db_path) as conn:
                    pd.read_csv(csv_file).to_sql(dataset_name, conn, if_exists='replace', index=False)
        except KeyError as e:
            print(e)
            pass


def check_sqlite_table_exists(db_path: str, table_name: str) -> bool:
    """
    Checks if a given table exists in the SQLite database.

    :param db_path: Path to the SQLite database file
    :param table_name: Name of the table to check
    :return: True if the table exists or DB is inaccessible, False otherwise
    """
    if not os.path.isfile(db_path):
        return True
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?;
        """, (table_name,))
        result = cursor.fetchone()
        return result is not None
    except sqlite3.Error as e:
        return True
    finally:
        if conn:
            conn.close()


def check_data():
    """
    Checks if the dataset table exists in the database.
    If not, it loads the dataset from the remote source.
    """
    if check_sqlite_table_exists(os.getenv("DB_PATH"), os.getenv("DATASET_NAME")):
        load_dataset(os.getenv("DATASET_URL"), os.getenv("DATASET_NAME"), os.getenv("DB_PATH"))


def update_env_variable(key, value):
    """
    Updates an environment variable in the .env file.

    :param key: The environment variable key
    :param value: The new value to set
    """
    load_dotenv(".env")
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            lines = f.readlines()
    else:
        lines = []
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            break
    with open(".env", "w") as f:
        f.writelines(lines)
    load_dotenv(override=True)
