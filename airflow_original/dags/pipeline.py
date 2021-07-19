from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.postgres.operators.postgres import PostgresOperator

from datetime import datetime, timedelta
import csv
import json
def write_data():
    with open('/opt/airflow/dags/files/write.csv', 'a') as f:
        f.write('write\n')

def read_data():
    with open('/opt/airflow/dags/files/read.csv', 'a') as f:
        f.write('read\n')


default_args = {
    "owner": "airflow",
    "email_on_failure": False,
    "email_on_retry": False,
    "email": "admin@localhost.com",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    'depend_on_past': False

}

with DAG("pipeline", start_date=datetime(2021, 7, 10),schedule_interval='@daily',
         default_args=default_args) as dag:

    web_scrape = BashOperator(
        task_id="web_scrape",
        bash_command="web.sh",

    )

    check_file = FileSensor(
        task_id="check_file",
        filepath="new.json",
        fs_conn_id="check_file",
        poke_interval=5
    )

    write_data = PythonOperator(
        task_id='write_data',
        python_callable=write_data
    )

    # read_data = PythonOperator(
    #     task_id='read_data',
    #     python_callable=read_data
    # )

    web_scrape >> check_file >> write_data