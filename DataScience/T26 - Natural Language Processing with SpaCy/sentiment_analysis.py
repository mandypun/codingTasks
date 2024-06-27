# Import necessary libraries
import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import textwrap

# Load the spaCy model
nlp = spacy.load('en_core_web_sm') 
nlp.add_pipe('spacytextblob')

# Load in the dataset
df = pd.read_csv('amazon_product_reviews.csv')

# Understanding the dataset
print('\n\n*************** Understand the dataset ***************')
print('df.info: ', df.info())
print('df.shape: ', df.shape)

# function for Text Preprocessing
def preprocess_text(text, filter_stopwords=True):
    # Create a spaCy document
    doc = nlp(text)
    
    # Remove stopwords (default option) and punctuation
    filtered_tokens = [token for token in doc if (not filter_stopwords or 
                                                  not token.is_stop) and not token.is_punct]
    
    # Lemmatize the tokens
    lemmatized_tokens = [token.lemma_ for token in filtered_tokens]
        
    # NER regconition
    print('NER          : ', [(i, i.label_, i.label) for i in doc.ents])

    # Join the lemmatized tokens back into a string
    preprocessed_text = ' '.join(lemmatized_tokens)
    print('Preprocessed : ', preprocessed_text)
    
    return preprocessed_text

# Preprocess the reviews.text column
print('\n\n*************** Preprocessing the sample data ***************')

# Remove missing values
# There is 1 missing value in reviews.text
df = df.dropna(subset=['reviews.text'])

# Convert the value to string to avoid error in analysis
df['preprocessed_text'] = df['reviews.text'].apply(str)

# Strip to remove leading space characters that reduces NER
df['preprocessed_text'] = df['preprocessed_text'].str.strip()

# Test the model on sample product reviews (first 60 reviews in the dataset).
# Use columns related to review only
# reviews_data = df.iloc[0:60,[11,14,16,17,20,21]]  
reviews_data = df.loc[0:60,['reviews.rating','reviews.text','preprocessed_text']]

# Apply SpaCy preprocess to review text
reviews_data['preprocessed_text'] = reviews_data['preprocessed_text'].apply(
    preprocess_text)

print('\n\n*************** Data after preprocessing ***************')
print(reviews_data.head())

# function for sentiment analysis
def analyze_sentiment(text):
    # Create a spaCy document
    doc = nlp(text)
    
    # Get the sentiment polarity and subjectivity
    
    # sentiment = doc.sentiment     # always 0
    polarity = doc._.blob.polarity
    subjectivity = doc._.blob.subjectivity
    sent = doc._.blob.sentiment

    # Determine the sentiment label
    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment, polarity

# Save sentiment and polarity in the dataset for further review
reviews_data['sentiment'], reviews_data['polarity'] = zip(
    *reviews_data['preprocessed_text'].apply(analyze_sentiment))

# View the analysis result
print('\n\n*************** Reviewing the analysis result (Sort by least polarity) ***************')
# Sort by least polarity
for index, row in reviews_data.sort_values('polarity').iterrows():
    print(f"index: {index}")
    print(textwrap.fill(f"Review: {row['reviews.text']}", 180))  # Wrap for long text
    print(f"Preprocessed Text: {row['preprocessed_text']}")
    print(f"Review Rating: {row['reviews.rating']}")   # For verifying sentiment result
    print(f"Sentiment: {row['sentiment']}")  # Result from the model
    print(f"Polarity: {row['polarity']}")  # Result from the model
    print()

# Further experiments on the model 
print('\n\n*************** Further experiments on the model ***************')

# Strip the text can help on NER
text = '        Google is  a good searching engine and fast to get result.'
print('Case 1 - text: ', text)
print('Result with strip():', analyze_sentiment(preprocess_text(text.strip())))
print('Result w/o  strip():', analyze_sentiment(preprocess_text(text)))  # Can't recongnise Google as ORG
print('Can\'t recongnise Google as ORG')

# The lower/upper cases of name may affect NER
text = 'This travel book introduces lots of great places in USA, and they are not well known by people.'
print('\nCase 2 - text: ', text)
print('Result with lower():', analyze_sentiment(preprocess_text(text.lower())))  # Can't recongnise USA
print('Result with upper():', analyze_sentiment(preprocess_text(text.upper())))
print('Result w/o changing case:', analyze_sentiment(preprocess_text(text)))
print('Can\'t recongnise USA')

# 'Not' is removed and causes wrong analysis result
text = 'Not easy for elderly users cease of ads that pop up.'
print('\nCase 3 - text: ', text)
print('Result with removing stopwords:', analyze_sentiment(preprocess_text(
    text, filter_stopwords=True)))  # Wronly interpreted as positive sentiment
print('Result w/o  removing stopwords:', analyze_sentiment(preprocess_text(
    text, filter_stopwords=False)))
print('\'Not\' is removed and causes wrong analysis result')