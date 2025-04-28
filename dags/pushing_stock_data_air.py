from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.decorators import task
# to push data into the postgres sql
from airflow.providers.mysql.hooks.mysql import MySqlHook

import json
from airflow.utils.dates import days_ago
import yfinance as yf
import pandas as pd
import pendulum

local_tz = pendulum.timezone("Asia/Dubai")

with DAG(
    dag_id="send_stockdata_database",
    start_date=pendulum.now("Asia/Dubai").subtract(days=1),  # Correct
    schedule_interval="0 0 * * 1-5",  # Every weekday (Mon-Fri) at 00:00 UTC
    catchup=False,
) as dag:

     # stock_data = yf.Ticker(tickers)

     # stock_history = stock_data.history(period="3mo")


     # print(stock_history)

     @task
     def create_table():
        
          # config = {
          #           'user': 'root',
          #           'password': 'root',
          #           'host': 'localhost',  # or your server IP
          #           'database': 'stocks',
          #           'raise_on_warnings': True
          #           }

          mysql_hook = MySqlHook(mysql_conn_id='mysql_stocks')
          
          create_table_query = '''
               CREATE TABLE IF NOT EXISTS stockdata (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    Open FLOAT,
                    High FLOAT,
                    Low FLOAT,
                    Close FLOAT,
                    Volume BIGINT,
                    Dividends FLOAT,
                    Stock_Splits FLOAT,
                    Date DATETIME,
                    ticker VARCHAR(10)
               )  '''
          
          mysql_hook.run(create_table_query)


     @task
     def fetch_and_transform_stock_data():
        
          tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL"]

          all_frames = []

          for ticker in tickers:

               stock = yf.Ticker(ticker)

               stock_history = stock.history(period="1d")

               stock_history = stock_history.reset_index()

               stock_history["ticker"] = ticker

               all_frames.append(stock_history)
     
          dataframe = pd.concat(all_frames,ignore_index=True)
        
          return dataframe
     

     
     @task
     def load_data_to_sql(transformed_data):

          records = list(transformed_data.itertuples(index=False, name=None))


          mysql_hook = MySqlHook(mysql_conn_id='mysql_stocks')

          # insert_query = '''
          #           INSERT INTO stockdata (
          #           Open, High, Low, Close, Volume, Dividends, Stock_Splits, Date, ticker
          #      ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
          # '''

          mysql_hook.insert_rows(
               table="StockData",
               rows=records,
               target_fields=[
                    "Date", "Open", "High",
                    "Low", "Close", "Volume", "Dividends", "Stock_Splits","ticker"
               ],
               commit_every=5,    # commit after these 5 rows
          )
          


     create_table() # Ensure table is created before extraction

     stock_data = fetch_and_transform_stock_data()  # Fetch stock data

     transformed_data = load_data_to_sql(stock_data)

    


