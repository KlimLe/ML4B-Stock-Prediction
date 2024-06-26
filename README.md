# Stock Prediction App

Antonios Xenidis (StudOn-Username/Enrollment Number)
Felix Röder (StudOn-Username/Enrollment Number)
Gabriela Lenhart (il14ysuq/21755242) 
Shu Qiu (StudOn-Username/Enrollment Number)

## 1. Motivation

Our primary goal is to generate a model to predict the future stock prices of multiple companies using a combination of historical stock prices, technical indicators, fundamental data, and sentiment and embeddings from financial news articles. This multi-faceted approach aims to capture various factors that influence stock prices, providing a more comprehensive and accurate prediction. 

## 2. Related Work
What have others done in your area of work/ to answer similar questions?
Discussing existing work in the context of your work

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

The model is a deep learning-based multi-output regression model designed to predict the future stock prices of multiple companies. It incorporates a range of features, including technical indicators, fundamental data, sentiment analysis, and text embeddings from news articles. Here's a detailed description of the model's components and workflow: 

#### Features

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


### 3.2 Advanced Model

Overview 

We added 3 other features to the basic model, in order to achieve advanced output.

#### Features

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
