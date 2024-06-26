# Stock Prediction App

Antonios Xenidis (StudOn-Username/Enrollment Number)
Felix Röder (StudOn-Username/Enrollment Number)
Gabriela Lenhart (il14ysuq/21755242) 
Shu Qiu (StudOn-Username/Enrollment Number)

## 1. Motivation

Our primary goal is to generate a model to predict the future stock prices of multiple companies using a combination of historical stock prices, technical indicators, fundamental data, and sentiment and embeddings from financial news articles. This multi-faceted approach aims to capture various factors that influence stock prices, providing a more comprehensive and accurate prediction. 

## 2. Related Work
**What have others done in your area of work/ to answer similar questions?**
**Discussing existing work in the context of your work**

## 3. Methodology

### 3.1 General Methodology
How did you proceed to achieve your project goals? 
Describe which steps you have undertaken
Aim: Others should understand your research process

### 3.2 Data Understanding and Preparation
Introduce the dataset to the reader
Describe structure and size of your dataset
Describe specialities
Describe how you prepare the dataset for your project

### 3.3 Modeling and Evaluation
Describe how you train your models
Describe how you evaluate your models/ which metrics you use

### 3.1 Basic Model

The model you've built is a deep learning-based multi-output regression model designed to predict the future stock prices of multiple companies. It incorporates a range of features, including technical indicators, fundamental data, sentiment analysis, and text embeddings from news articles. Here's a detailed description of the model's components and workflow: 
Describe the model architecture(s) you selected

Components 

Data Collection and Preprocessing: 

Financial News Dataset: The dataset consists of news articles related to various companies, along with their publication dates. 

Stock Prices and Fundamental Data: Stock prices and fundamental data (like PE Ratio, EPS, Revenue, and Market Cap) are fetched for specific companies using the yfinance library. 

Text Preprocessing: 

Tokenization and Lemmatization: The news articles are preprocessed to remove non-alphabetical characters, convert text to lowercase, remove stop words, and lemmatize the words to their base form. 

Sentiment Analysis: The sentiment polarity of each news article is calculated using the TextBlob library. 

Text Embeddings: Each preprocessed news article is converted into a numerical vector (embedding) using a pre-trained BERT model from the transformers library. 

Technical and Fundamental Data Processing: 

Technical Indicators: Technical indicators are computed from the stock price data using the ta library. 

Handling Missing Values: Missing values in technical indicators and fundamental data are imputed using the KNN imputer. 

Data Aggregation and Integration: 

Aggregate News Data: The news data is aggregated daily to compute the average sentiment and BERT embeddings. 

Merge with Stock Data: The aggregated news data is merged with the stock price data and fundamental data to create a comprehensive dataset for each company. 

Sequence Creation: 

Look-Back Window: For each company, sequences of data (including stock prices, sentiment scores, text embeddings, technical indicators, and fundamental data) are created using a defined look-back window. This means that for each prediction, the model looks at a sequence of previous days' data. 

Future Price as Target: The future stock price (the price on the next day) is used as the target variable. 

Model Architecture: 

Input Layer: The input layer accepts sequences of combined features for each company. 

Transformer Block: A custom Transformer block is used to process the sequential data. This block includes multi-head attention mechanisms and feed-forward neural networks, along with layer normalization and dropout for regularization. 

Global Average Pooling: The output of the Transformer block is globally averaged to create a fixed-size representation. 

Dense Layers: A dense layer with batch normalization and dropout is used for further processing. 

Output Layers: Separate output layers for each company's stock price prediction, each producing a single value (the predicted future price). 

Training and Evaluation: 

Data Splitting: The data is split into training and validation sets using a time-series split to maintain temporal order. 

Model Compilation and Training: The model is compiled with mean squared error (MSE) loss functions for each company's output and trained using the Adam optimizer. Early stopping is used to prevent overfitting. 

Prediction and Inverse Transformation: After training, the model makes predictions on the validation set, and these predictions are inverse-transformed to their original scale using the previously fitted scalers. 

Saving the Model: 

Model Persistence: The trained model is saved to disk for future use. 

### 3.2 Advanced Model

Overview 

This model is designed to predict future stock prices for selected companies based on a combination of historical stock prices, technical indicators, fundamental data, and news articles. The model leverages advanced natural language processing (NLP) techniques to analyze news articles and extract valuable features, such as sentiment, topics, TF-IDF vectors, and named entities. These features, along with stock-related data, are fed into a transformer-based neural network model to make predictions. 

Data Preparation 

Load and Preprocess News Articles: 

The news dataset is loaded and preprocessed. This includes cleaning the text, removing special characters, converting to lowercase, and lemmatizing words. 

Sentiment analysis is performed on the preprocessed articles using TextBlob to derive a sentiment score. 

Named Entity Recognition (NER) is performed using SpaCy to extract entities from the articles. 

TF-IDF Vectorization and Topic Modeling: 

TF-IDF vectorization is applied to the articles to convert them into numerical vectors representing the importance of each word in the document. 

Latent Dirichlet Allocation (LDA) is used for topic modeling to identify the main topics in the articles. 

BERT Embeddings: 

BERT embeddings are generated for the articles using the RoBERTa model. These embeddings capture the contextual meaning of the text. 

NER Vectorization: 

The extracted entities are converted into a fixed-size vector representing the frequency of each entity type. 

Fetch Stock Data and Fundamentals: 

Historical stock prices are fetched for each company from Yahoo Finance. Technical indicators are calculated and missing values are imputed. 

Fundamental data such as PE ratio, EPS, revenue, and market cap are also fetched. 

Aggregate and Merge Data: 

News data is aggregated by day and merged with the stock data. Missing dates in the news data are filled with neutral values. 

Separate aggregations are done for company-specific news and general news. 

Sequence Creation 

Create Sequences: 

For each company, sequences of fixed length (look_back days) are created. Each sequence includes: 

News embeddings (company-specific and general) 

Sentiment scores (company-specific and general) 

Technical indicators 

Fundamental data 

Topic vectors 

TF-IDF vectors 

NER vectors (company-specific and general) 

The target for each sequence is the stock price on the next day. 

Normalize Features: 

The combined features are scaled using StandardScaler. 

The target prices are also scaled individually for each company. 

Model Architecture 

Transformer-Based Model: 

The model uses a transformer block to process the sequences. The transformer block includes: 

Multi-head self-attention mechanism to capture dependencies between different time steps. 

Feed-forward neural network layers. 

Layer normalization and dropout for regularization. 

Global average pooling is applied to the output of the transformer block. 

A dense layer with batch normalization and dropout is added for further processing. 

Output Layers: 

Separate output layers (dense layers) are created for each company to predict their respective stock prices. 

Compile Model: 

The model is compiled using mean squared error (MSE) as the loss function for each company's output. 

Training and Prediction 

Train Model: 

The model is trained using the prepared sequences and targets. Early stopping is used to prevent overfitting. 

Make Predictions: 

Predictions are made on the validation data. 

The predicted prices are inverse-transformed to get them back to the original scale. 

Save Model: 

The trained model is saved for future use. 

Summary 

This model combines advanced NLP techniques with time series forecasting to predict stock prices based on a comprehensive set of features extracted from news articles and stock data. By leveraging transformer-based neural networks, the model can capture complex dependencies and provide accurate predictions for multiple companies simultaneously. 

## 4. Results
Describe what artifacts you have build
Describe the libraries and tools you use
Describe the concept of your app
Describe the results you achieve by applying your trained models on unseen data
Descriptive Language (no judgement, no discussion in this section -> just show what you built)

## 5. Discussion
Now its time to discuss your results/ artifacts/ app 
Show the limitations : e.g. missing data, limited training ressources/ GPU availability in Colab, limitaitons of the app
Discuss your work from an ethics perspective:
Dangers of the application of your work (for example discrimination through ML models)
Transparency 
Effects on Climate Change 
Possible sources  Have a look at the "Automating Society Report";  Have a look at this website and their publications
Further Research: What could be next steps for other researchers (specific research questions)

## 6. Conclusion
Short summary of your findings and outlook
