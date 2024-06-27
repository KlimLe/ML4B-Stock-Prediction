# Stock Prediction App

Antonios Xenidis (Up13ohij/23142875)  
Felix Röder (na60puza/23014836)  
Gabriela Lenhart (il14ysuq/21755242)  
Shu Qiu (es36ydyh/22732779)  

## 1. Motivation

Our primary goal is to generate a model to predict the future stock prices of multiple companies using a combination of historical stock prices, technical indicators, fundamental data, and sentiment and embeddings from financial news articles. This multi-faceted approach aims to capture various factors that influence stock prices, providing a more comprehensive and accurate prediction. 

### Business Case
Our goal is to enable informed decisions for investors. This requires reliable predictions in order to minimize risks and maximize returns.

#### Challenges
- The Complexity and volatility of financial markets makes accurate predictions challenging.
- Choosing and combining the right model components is crucial for accurate forecasts.
- Selecting, updating, and processing a robust dataset is essential for model accuracy.

## 2. Related Work
We started our coding-journey by exploring the internet, searching for approaches to this matter. 
You can find many articles and  [youtube](https://www.youtube.com/results?search_query=Stock+prediction+machine+learning) videos describing various approaches to this specific goal, which we looked through in order to find inspiration on how to address our goal.

## 3. Methodology

### 3.1 General Methodology
We started by reviewing the current approaches to predicting stock prices using machine learning algorithms, with the aim of establishing a basic framework. The next step was to find suitable datasets that we could use to train the model and find a way to continuously update the data. The following steps consisted mainly of continuously improving the current model to make it more powerful and thus produce better predictions. 

### 3.2 Data Understanding and Preparation
https://www.kaggle.com/datasets/leukipp/reddit-finance-data
We used the dataset included in the link, sorting it by date and using only the headlines associated with Apple, Amazon or Google.

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

## 4. Results
Describe what artifacts you have build
Describe the libraries and tools you use
Describe the concept of your app
Describe the results you achieve by applying your trained models on unseen data
Descriptive Language (no judgement, no discussion in this section -> just show what you built)

## 5. Discussion

Overall, we managed to build two working models and an interactive streamlit-app, to demonstrate our work. Although our work fits with the business case, there were some obstacles and limitations to our results.

#### 5.1 Limited Access to suitable Data
We had difficulty obtaining appropriate datasets, and the search was time-consuming, especially due to scaling. To acces live news, we built a web scraper, however it unfortunately failed at paywalls.

#### 5.2 Libraries
When using Streamlit, incompatible versions of the libraries used in the Streamlit environment caused errors.

#### 5.3 Model
- Automated, repeated retraining of the model was abandoned due to poor data quality.
- Tuning of hyperparameters failed because of an unsuitable data structure.
- Our acvanced model shows better performance at the cost of time and storage space.
- We are experiencing overfitting due to having a small dataset.

#### 5.4 CPU
Our advanced model is particularly demanding, which is why we moved our development to Google Colab.

#### 5.5 Repositories
The transfer of code from Google Colab to GitHub caused errors, and debugging is still incomplete.

#### 5.6 Ethics
- Ethical assesment should explore whether the use of prediction models could potentially destabilize financial markets. Widespread use among market participants could lead to self-fulfilling prophecies and increased market volatility.
- Whilst using Deep Learning models there exists the black box problem, which refers to low transparency of decisions of the models and therefore low transparency for the users.

## 6. Conclusion
We are happy to introduce two working models which are strongly oriented towards our business case. 
With the opportunity to obtain more data, there would be a possibility to overcome Overfitting and an overall stronger performance. 
