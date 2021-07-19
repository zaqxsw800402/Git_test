import datetime

from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
from datetime import timedelta
default_args = {
    "owner": "airflow",
    "email_on_failure": False,
    "email_on_retry": False,
    "email": "admin@localhost.com",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    'depend_on_past': False

}

# create_pet_table, populate_pet_table, get_all_pets, and get_birth_date are examples of tasks created by
# instantiating the Postgres Operator


def load_logs():
    conn = PostgresHook(postgres_conn_id='postgres_default').get_conn()
    cur = conn.cursor()
    SQL_STATEMENT = """
        COPY newbc (title, price, stock)
        FROM STDIN WITH CSV HEADER
        """
    # SQL_STATEMENT = """
    #     COPY books FROM STDIN WITH DELIMITER AS E'\n'
    #     """

    with open('/opt/webscrape/bookstoscrape/new.csv', 'r') as f:

        cur.copy_expert(SQL_STATEMENT, f)
        conn.commit()


with DAG(
    dag_id="postgres_operator_dag",
    start_date=datetime.datetime(2020, 2, 2),
    schedule_interval="@once",
    default_args=default_args,
    catchup=False,
) as dag:

    create_pet_table = PostgresOperator(
        task_id="create_pet_table",
        postgres_conn_id="postgres_default",
        sql="""
                CREATE TABLE IF NOT EXISTS newbc (
                books_id SERIAL PRIMARY KEY,
                title VARCHAR NOT NULL,
                price VARCHAR NOT NULL,
                stock VARCHAR NOT NULL);
              """,)
    copy_data = PythonOperator(
        task_id='copy_data',
        # postgres_conn_id="postgres_default",
        python_callable=load_logs
    )

    select_table = PostgresOperator(
        task_id="select_table",
        postgres_conn_id="postgres_default",
        sql="""
                SELECT * FROM newb;
              """,)

    create_pet_table >> copy_data >> select_table