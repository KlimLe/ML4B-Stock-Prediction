# Stock Prediction App

Antonios Xenidis (Up13ohij/23142875)  
Felix Röder (na60puza/23014836)  
Gabriela Lenhart (il14ysuq/21755242)  
Shu Qiu (es36ydyh/22732779)  

## 1. Motivation
Our primary goal is to generate a model to predict the future stock prices of multiple companies using a combination of historical stock prices, technical indicators, fundamental data, and sentiment and embeddings from financial news articles. This multi-faceted approach aims to capture various factors that influence stock prices, providing a more comprehensive and accurate prediction. 

#### Revolutionizing Stock Prediction: Leveraging BERT for Deep Financial News Analysis
Traditional sentiment analysis for stock prediction often only relies on simplistic sentiment scores derived from news headlines or social media posts. While these methods provide a rudimentary gauge of market sentiment, they lack the depth and contextual understanding needed for precise predictions.

Our approach leverages the power of Bidirectional Encoder Representations from Transformers (BERT) to process and understand the nuances of financial news and its impact on stock movements. By training our model on historical data, including how stock prices have reacted to news in the past, we can identify complex patterns and relationships that are often missed by traditional sentiment analysis.

### Business Case
Our goal is to enable informed decisions for investors. This requires reliable predictions in order to minimize risks and maximize returns.

#### Challenges
- The Complexity and volatility of financial markets makes accurate predictions challenging.
- Choosing and combining the right model components is crucial for accurate forecasts.
- Selecting, updating, and processing a robust dataset is essential for model accuracy.

## 2. Related Work
We first conducted an extensive review of various machine learning models to determine the most suitable architecture for our stock price prediction project. Our focus was on evaluating Long Short-Term Memory (LSTM) networks and Transformer architectures, both of which have shown promise in time series forecasting. After a detailed comparative analysis, we decided to utilize the Transformer architecture due to its superior performance.
Transformers, initially introduced by Vaswani et al. (2017) for natural language processing tasks, have recently been adapted for time series prediction due to their ability to handle long-range dependencies more efficiently than RNN-based models. Unlike LSTMs, which process data sequentially, Transformers utilize a self-attention mechanism that allows for parallel processing and better captures relationships in the data.

Several key papers provided comprehensive insights into the application of Transformers for stock price prediction:
- Hasan et al. (2023) focused on automated sentiment analysis for web-based stock and cryptocurrency news summarization using Transformer-based models. This study illustrated the effectiveness of models like BERT and XLNET in extracting and leveraging sentiment from financial news for stock price prediction.
- Daiya and Lin (2021) presented a model for stock movement prediction and portfolio management using multimodal learning with Transformers. This work demonstrated how integrating stock data and news can enhance prediction accuracy, leveraging the strengths of the Transformer architecture.
- Zhang et al. (2022) developed a Transformer-based attention network for stock movement prediction. The paper provided evidence of the model's ability to process and interpret news headlines and historical stock prices, achieving high prediction accuracy. 

We started our coding-journey by exploring the internet, searching for approaches to this matter. 
You can find many articles and  [youtube](https://www.youtube.com/results?search_query=Stock+prediction+machine+learning) videos describing various approaches to this specific goal, which we looked through in order to find inspiration on how to address our goal.

## 3. Methodology

### 3.1 General Methodology
We started by reviewing the current approaches to predicting stock prices using machine learning algorithms, with the aim of establishing a basic framework. The next step was to find suitable datasets that we could use to train the model and find a way to continuously update the data. The following steps consisted mainly of continuously improving the current model to make it more powerful and thus produce better predictions. 

### 3.2 Data Understanding and Preparation
https://www.kaggle.com/datasets/leukipp/reddit-finance-data
We used the dataset included in the link, sorted it by date and used only the headlines associated with Apple, Amazon or Google.

### 3.3 Modeling and Evaluation

#### 3.1 Basic Model

The model is a deep learning-based multi-output regression model designed to predict the future stock prices of multiple companies. It incorporates a range of features, including technical indicators, fundamental data, sentiment analysis, and text embeddings from news articles. Here's a detailed description of the model's components and workflow.

##### 1. Data Collection and Preprocessing: 

 - **Financial News Dataset**: The dataset consists of news articles related to various companies, along with their publication dates. 
 - **Stock Prices and Fundamental Data**: Stock prices and fundamental data (like PE Ratio, EPS, Revenue, and Market Cap) are fetched for specific companies using the yfinance library. 

##### 2. Text Preprocessing: 

- **Tokenization and Lemmatization**: The news articles are preprocessed to remove non-alphabetical characters, convert text to lowercase, remove stop words, and lemmatize the words to their base form. 

 - **Sentiment Analysis**: The sentiment polarity of each news article is calculated using the TextBlob library. 

 - **Text Embeddings**: Each preprocessed news article is converted into a numerical vector (embedding) using a pre-trained BERT model from the transformers library. 

##### 3. Technical and Fundamental Data Processing: 

 - **Technical Indicators**: Technical indicators are computed from the stock price data using the ta library. 

 - **Handling Missing Values**: Missing values in technical indicators and fundamental data are imputed using the KNN imputer. 

##### 4. Data Aggregation and Integration: 

 - **Aggregate News Data**: The news data is aggregated daily to compute the average sentiment and BERT embeddings. 

 - **Merge with Stock Data**: The aggregated news data is merged with the stock price data and fundamental data to create a comprehensive dataset for each company. 

##### 5. Sequence Creation: 

 - **Look-Back Window**: For each company, sequences of data (including stock prices, sentiment scores, text embeddings, technical indicators, and fundamental data) are created using a defined look-back window. This means that for each prediction, the model looks at a sequence of previous days' data. 

 - **Future Price as Target**: The future stock price (the price on the next day) is used as the target variable. 

##### 6. Model Architecture: 

 - **Input Layer**: The input layer accepts sequences of combined features for each company. 

 - **Transformer Block**: A custom Transformer block is used to process the sequential data. This block includes multi-head attention mechanisms and feed-forward neural networks, along with layer normalization and dropout for regularization. 

- **Global Average Pooling**: The output of the Transformer block is globally averaged to create a fixed-size representation. 

- **Dense Layers**: A dense layer with batch normalization and dropout is used for further processing. 

 - **Output Layers**: Separate output layers for each company's stock price prediction, each producing a single value (the predicted future price). 

##### 7. Training and Evaluation: 

 - **Data Splitting**: The data is split into training and validation sets using a time-series split to maintain temporal order. 

- **Model Compilation and Training**: The model is compiled with mean squared error (MSE) loss functions for each company's output and trained using the Adam optimizer. Early stopping is used to prevent overfitting. 

- **Prediction and Inverse Transformation**: After training, the model makes predictions on the validation set, and these predictions are inverse-transformed to their original scale using the previously fitted scalers. 

##### 8. Saving the Model


#### 3.2 Advanced Model


We added following other features to the basic model, in order to achieve advanced output:

##### 1. Named Entity Recognition (NER):
Named Entity Recognition (NER) is performed using SpaCy to extract entities from the articles. 

##### 2. TF-IDF Vectorization and Topic Modeling: 
TF-IDF vectorization is applied to the articles to convert them into numerical vectors representing the importance of each word in the document. 
 
##### 3. Topic Modelling: 
Latent Dirichlet Allocation (LDA) is used for topic modeling to identify the main topics in the articles.

## 4. Discussion

Overall, we managed to build two working models and an interactive streamlit-app, to demonstrate our work. Although our work fits with the business case, there were some obstacles and limitations to our results.

#### 4.1 Limited Access to suitable Data
We had difficulty obtaining appropriate datasets, and the search was time-consuming, especially due to scaling. To acces live news, we built a web scraper, however it unfortunately failed at paywalls.

#### 4.2 Libraries
When using Streamlit, incompatible versions of the libraries used in the Streamlit environment caused errors.

#### 4.3 Model
- Automated, repeated retraining of the model was abandoned due to poor data quality.
- Tuning of hyperparameters failed because of an unsuitable data structure.
- Our acvanced model shows better performance at the cost of time and storage space.
- We are experiencing overfitting due to having a small dataset.

#### 4.4 CPU
Our advanced model is particularly demanding, which is why we moved our development to Google Colab.

#### 4.5 Repositories and Streamlit
By utilizing Colab, we can ensure a higher level of stability for running our Streamlit app. The transfer of code from Google Colab to GitHub caused errors, and debugging was successful to the point where we were able to run the Streamlit app. However, Streamlit Cloud terminates the running app due to a lack of RAM in its environment.

We tried the following measures:
- We added an instruction to prevent the model from being downloaded more than once, as this issue occurred previously.
- Given that BERT embeddings are memory-intensive, we implemented caching to avoid unnecessary multiple generations of these embeddings.

Unfortunately, these measures did not resolve the issue, likely because the RoBERTa model requires significant memory.

However, to be able to present our Streamlit-App, we prepared a colab-file and a manual. (See 6.)

#### 4.6 Ethics
- Ethical assesment should explore whether the use of prediction models could potentially destabilize financial markets. Widespread use among market participants could lead to self-fulfilling prophecies and increased market volatility.
- Whilst using Deep Learning models there exists the black box problem, which refers to low transparency of decisions of the models and therefore low transparency for the users.

## 5. Conclusion
We are happy to introduce two working models which are strongly oriented towards our business case. 
With the opportunity to obtain more data, there would be a possibility to overcome Overfitting and an overall stronger performance. 



## 6. Repository Description and Colab-manual
- AdvancedModel.ipynb: This shows our colab-code, which we used to train our advanced model
- BasicModel.ipynb: This shows our colab-code, which we used to train our basic model, saving it, and deploying the streamlit-app via pygrok
- BasicModelTrainer.py: The code for training our basic model
- app.py: The Code for our Streamlit-App with changes regarding the mentioned RAM-issue (see 4.5), debugging incomplete
- final_dataset.csv: Our Training-Dataset
- final_dataset_without_last_column.csv: Our Prediction-Dataset
- WebScraper.py: Our WebScraper
  

### Colab-manual
The [https://colab.research.google.com/drive/1tWe3-ttLCqmtqMTxYmETmppL8rc_okQD?usp=sharing] file contains two cells. The first cell creates a streamlit app using the %%writefile app.py command. To deploy the app, follow these steps:
- Run the first cell to write the Streamlit app code to app.py.
- Run the second cell to start the deployment process. This cell ensures that all tunnels are properly set up. After running this cell, click the HTTPS link that appears to access your Streamlit app.

If you encounter any issues, such as connection errors or other types of errors, follow these steps:
- Rerun the second cell. This will kill any existing tunnels and attempt to re-establish a new connection.
- If the problem persists, rerun the second cell again until the connection stabilizes and the app runs smoothly.

