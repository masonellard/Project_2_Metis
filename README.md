# Predicting Domestic Gross Movie Sale With Data Scraped From IMDb
In this project, I scraped data on approximately 2400 movies from IMDb, and compared prediction results between OLS, Lasso, Ridge, K-Nearest Neighbors, and XGBoost.
I used the following features in my models: budget, runtime, days since release, genre, mpaa rating, director rank, writer rank, and cast rank.

## File Descriptions

### [IMDb_scrape.ipynb](https://github.com/masonellard/Project_2_Metis/IMDb_scrape.ipynb)
In this file, I scraped my intitial uncleaned data from IMDb.com using pandas and BeautifulSoup. The resulting dataframe was exported to imdb_data.csv.

### [data_cleaning_and_engineering.ipynb](https://github.com/masonellard/Project_2_Metis/data_cleaning_and_engineering.ipynb)
I imported the uncleaned csv file, and did some initial cleaning and feature engineering. Feature engineering includes turning genre and mpaa rating into dummies, and transforming director, writers, and cast into continous variables. The cleaned dataframe is exported to imdb_data_cleaned.csv.

### [model_fitting.ipynb](https://github.com/masonellard/Project_2_Metis/model_fitting.ipynb)
I import the cleaned dataset and fit and tune 5 models: OLS, Lasso, Ridge, K-Nearest Neighbors, and XGBoost. I export my best model (XGBoost) into a pickle object for use in the prediction [app](https://github.com/masonellard/Project_2_Metis/tree/main/prediction_app)

### App Instructions:
Download all files in the [app folder](https://github.com/masonellard/Project_2_Metis/tree/main/prediction_app). If you don't already have streamlit, use 'pip install streamlit' in the command line. Once you have streamlit installed, navigate to the directory holding the files from the app folder, and run the command: streamlit run imdb_predict_app.py 
