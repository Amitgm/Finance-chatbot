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
from datetime import datetime
from transformers import pipeline

local_tz = pendulum.timezone("Asia/Dubai")

with DAG(
    dag_id="send_sentiment_database",
    start_date=pendulum.now("Asia/Dubai").subtract(days=1),  # Correct
    schedule_interval="0 0 * * 1-5",  # Every weekday (Mon-Fri) at 00:00 UTC
    catchup=False,
) as dag:

     # stock_data = yf.Ticker(tickers)

     # stock_history = stock_data.history(period="3mo")


     # print(stock_history)

    @task
    def create_table():
        

          mysql_hook = MySqlHook(mysql_conn_id='mysql_sentiments')
          
          create_table_query = '''
            CREATE TABLE IF NOT EXISTS SentimentNews ( id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT,
                    pubDate DATETIME,
                    sentiment VARCHAR(20)
               )  '''
          
          mysql_hook.run(create_table_query)


    @task
    def fetch_and_transform_sentiment_data():
          
        sentiment_data = []

        pipe = pipeline("text-classification", model="ProsusAI/finbert")

        tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL"]

        keys = ["title","summary","pubDate","sentiment"]

        for symbol in tickers:

            ticker = yf.Ticker(symbol)

            news_list = ticker.news

            entry = {"symbol": symbol, "news": []}

            # goin through list of news, taking out each news
            for news in news_list:
                # taking out the news content
                content = news["content"]

                content_dict = {}

                for key in keys:
            
                    # getting the senitments out
                    if key == "sentiment":

                        value = pipe(content.get("summary", None))[0]["label"]

                    else:

                        value = content.get(key, None)

                    # if value is not None
                    if value is not None:

                        content_dict[key] = value        

                entry["news"].append(

                    content_dict
                )
                    
            sentiment_data.append(entry)

        return sentiment_data

    @task
    def load_data_to_sql(transformed_data):

        records = []
        for stock in transformed_data:

            symbol = stock['symbol']

            for news_item in stock['news']:

                title = news_item['title']

                summary = news_item['summary']

                pub_date = news_item['pubDate']

                pub_date = datetime.strptime(pub_date, '%Y-%m-%dT%H:%M:%SZ')

                sentiment = news_item['sentiment']
                
                records.append((symbol, title, summary, pub_date, sentiment))


        mysql_hook = MySqlHook(mysql_conn_id='mysql_sentiments')

        mysql_hook.insert_rows(

            table="SentimentNews",

            rows=records,

            target_fields=[
                
                "symbol", "title", "summary",
                "pubDate", "sentiment"
            ],

            commit_every=10,    # commit after these 5 rows
        )
        

    create_table() # Ensure table is created before extraction

    stock_data = fetch_and_transform_sentiment_data()  # Fetch stock data

    transformed_data = load_data_to_sql(stock_data)

    


