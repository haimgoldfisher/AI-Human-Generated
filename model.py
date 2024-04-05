import pickle
from collections import Counter
import numpy as np
import pandas as pd

# For NLP
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy

nlp = spacy.load("en_core_web_sm")
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

def add_features(data_to_modify):
    """
    Function to add features to the dataset
    """
    # utils lists
    sia = SentimentIntensityAnalyzer()
    punctuation_types = ['.', ',', "'", '"', '-', '?', ':', ')', '(', '!', '/', ';', '_', ']', '[']
    subordinating_conjunctions = ["after",
      "although",
      "as",
      "as far as",
      "as if",
      "as long as",
      "as soon as",
      "as though",
      "because",
      "before",
      "even if",
      "even though",
      "if",
      "if only",
      "if when",
      "inasmuch as",
      "in order that",
      "now that",
      "once",
      "provided that",
      "rather than",
      "since",
      "so that",
      "supposing",
      "than",
      "though",
      "till",
      "unless",
      "until",
      "when",
      "whenever",
      "where",
      "whereas",
      "wherever",
      "whether",
      "while",
      "why"]

    data = data_to_modify.copy()

    # functions
    def count_punctuation_row(text):
      """
      Function to compute punctuation ratio in text
      """
      count = sum(char in punctuation_types for char in text)
      return round(count/len(text), 3)*100

    def evaluate_tone(text):
      """
      Evaluate the tone of a text and return a score indicating the presence of emotion/personality.
      Higher scores suggest a more emotional or personalized tone, while lower scores suggest a robotic or flat tone.
      """
      # Analyze sentiment of the text
      sentiment_scores = sia.polarity_scores(text)

      # Calculate a score indicating the presence of emotion/personality
      tone_score = sentiment_scores['compound']

      return tone_score

    def get_sentiment(text):
      """
      This function takes a text as input and returns a dictionary containing the sentiment scores for the text.
      """
      sentiment_scores = sia.polarity_scores(text)
      return sentiment_scores

    def divide_sentiment_to_categories(data):
      longest_sentiment_dictionary = 0
      sentiments = []

      for row in data['Sentiment']:
        if len(row) > longest_sentiment_dictionary:
          longest_sentiment_dictionary = len(row)
          sentiments = row.keys()

      # add each sentiment section to the data frame as a column
      sentiments_dictionaries = [row for row in data['Sentiment']]
      sentiments_data = pd.DataFrame(data=sentiments_dictionaries)
      data = pd.concat([data, sentiments_data], axis=1)
      data.drop(['Sentiment'], axis=1, inplace=True)

      return data

    def richness(text):
      """
      This function takes a text as input and returns the richness of the text.
      """
      tokens = word_tokenize(text)
      unique_tokens = set(tokens)
      richness = len(unique_tokens) / len(tokens)
      return richness

    def count_char_percent(text, char):
      """
      This function takes a text and a character as input and returns the percentage of the character in the text.
      """
      count = text.count(char)
      percent = count / len(text) * 100
      return percent

    def get_pos_tags(text):
        """
        This function returns a dictionary of the percentages of each part of speech in a given text.
        """
        pos_tags = nltk.pos_tag(word_tokenize(text))
        counts = Counter(tag for word, tag in pos_tags)
        total_words = len(pos_tags)

        # Check if total_words is zero to avoid division by zero
        if total_words == 0:
            return {
                "Nouns": 0,
                "Pronouns": 0,
                "Verbs": 0,
                "Adjectives": 0,
                "Adverbs": 0,
                "Prepositions": 0,
                "Conjunctions": 0,
                "Interjections": 0
            }

        pos_percentages = {
            "Nouns": (counts["NN"] + counts["NNS"]) / total_words,
            "Pronouns": (counts["PRP"] + counts["PRP$"]) / total_words,
            "Verbs": (counts["VB"] + counts["VBD"] + counts["VBG"] + counts["VBN"] + counts["VBP"] + counts["VBZ"]) / total_words,
            "Adjectives": (counts["JJ"] + counts["JJR"] + counts["JJS"]) / total_words,
            "Adverbs": (counts["RB"] + counts["RBR"] + counts["RBS"]) / total_words,
            "Prepositions": (counts["IN"]) / total_words,
            "Conjunctions": (counts["CC"]) / total_words,
            "Interjections": (counts["UH"]) / total_words,
        }
        return pos_percentages

    def divide_pos_to_categories(data):
      longest_pos_dictionary = 0
      pos = []

      for row in data['POS_tags']:
        if len(row) > longest_pos_dictionary:
          longest_pos_dictionary = len(row)
          pos = row.keys()

      # add each sentiment section to the data frame as a column
      pos_dictionaries = [row for row in data['POS_tags']]
      pos_data = pd.DataFrame(data=pos_dictionaries)
      data = pd.concat([data, pos_data], axis=1)
      data.drop(['POS_tags'], axis=1, inplace=True)

      return data

    def get_entities(text):
      entities = nlp(text).ents
      return entities

    def entities_text_length_ratio(entities, text):
      return len(entities) / len(text)

    def get_stopwords_percent(text):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = word_tokenize(text)
        stopwords_count = sum(1 for word in words if word in stop_words)
        total_words = len(words)
        stopwords_percent = stopwords_count / total_words * 100
        return stopwords_percent

    def get_uppercase_percent(text):
        uppercase_count = sum(1 for char in text if char.isupper())
        total_chars = len(text)
        uppercase_percent = uppercase_count / total_chars * 100
        return uppercase_percent

    # Flesch-Kincaid Grade Level or Gunning Fog Index, which estimate the readability level of the text

    def flesch_reading_ease(text):
      """
      This function calculates the Flesch Reading Ease score for a given text.
      """
      # Count the number of words, sentences, and syllables in the text.
      words = text.split()
      sentences = text.count('.') + text.count('?') + text.count('!')
      syllables = sum(sum(1 for c in word if c.lower() in 'aeiou') for word in words)

      # Calculate the Flesch Reading Ease score.
      if sentences == 0:
        sentences = 1
      score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
      return score

    def flesch_kincaid_grade_level(text):
      """
      This function calculates the Flesch-Kincaid Grade Level for a given text.
      """
      # Calculate the Flesch Reading Ease score.
      score = flesch_reading_ease(text)

      # Convert the score to a grade level.
      grade_level = (0.39 * score) + 11.8
      return grade_level

    def gunning_fog_index(text):
      """
      This function calculates the Gunning Fog Index for a given text.
      """
      # Count the number of words, sentences, and complex words in the text.
      words = text.split()
      sentences = text.count('.') + text.count('?') + text.count('!')
      complex_words = sum(1 for word in words if len(word) >= 3 and sum(1 for c in word if c.lower() in 'aeiou') > 2)

      # Calculate the Gunning Fog Index.
      if sentences == 0:
        sentences = 1

      score = 0.4 * ((len(words) / sentences) + 100 * (complex_words / len(words)))
      return score

    def count_subordinate_clauses(text):
      """
      This function counts the number of subordinate clauses in a given text.
      """
      subordinate_clauses = 0
      for sentence in sent_tokenize(text):
        for word in sentence.split():
          if word in subordinating_conjunctions:
            subordinate_clauses += 1

      return subordinate_clauses

    def count_independent_clauses(text):
      """
      This function counts the number of independent clauses in a given text.
      """
      independent_clauses = 0
      for sentence in sent_tokenize(text):
        if not any(word in subordinating_conjunctions for word in sentence.split()):
          independent_clauses += 1
      return independent_clauses

    def syntactic_complexity(text):
      """
      This function calculates the syntactic complexity of a given text.
      """
      # Count the number of subordinate and independent clauses.
      subordinate_clauses = count_subordinate_clauses(text)
      independent_clauses = count_independent_clauses(text)

      # Calculate the syntactic complexity.
      if independent_clauses == 0:
        return 0
      else:
        return subordinate_clauses / independent_clauses

    def average_sentence_length(text):
        """
        This function calculates the average sentence length for a given text.
        """
        sentences = sent_tokenize(text)
        words_per_sentence = [len(sentence.split()) for sentence in sentences]
        average_length = sum(words_per_sentence) / len(sentences)
        return average_length



    # Punctuation precent
    data['punctuation%'] = data['text'].apply(lambda row: count_punctuation_row(row))
    # Tone
    data['tone'] = data['text'].apply(lambda row: evaluate_tone(row))
    # Sentiment
    data['Sentiment'] = data['text'].apply(get_sentiment)
    data = divide_sentiment_to_categories(data)
    # Richness
    data['Richness'] = data['text'].apply(richness)
    # features of '.' , ',' and '?'
    data['%_comma'] = data['text'].apply(lambda x: count_char_percent(x, ','))
    data['%_period'] = data['text'].apply(lambda x: count_char_percent(x, '.'))
    data['%_q_mark'] = data['text'].apply(lambda x: count_char_percent(x, '?'))
    # POS
    data['POS_tags'] = data['text'].apply(get_pos_tags)
    data = divide_pos_to_categories(data)
    # Entities
    data['Entities'] = data['text'].apply(get_entities)
    # Entities ratio
    data['Entities_ratio'] = data.apply(lambda row: entities_text_length_ratio(row['Entities'], row['text']), axis=1)
    data.drop('Entities', axis=1, inplace=True)
    # stop words
    data['%_stopwords'] = data['text'].apply(get_stopwords_percent)
    # upper case
    data['%_uppercase'] = data['text'].apply(get_uppercase_percent)
    # test_length
    data['text_length'] = data['text'].apply(lambda x: len(x))
    # num words
    data['num_words'] = data['text'].apply(lambda x: len(x.split()))
    # Flesch-Kincaid Grade Level or Gunning Fog Index
    data['FleschReadingEase'] = data['text'].apply(flesch_reading_ease)
    data['FleschKincaidGradeLevel'] = data['text'].apply(flesch_kincaid_grade_level)
    data['GunningFogIndex'] = data['text'].apply(gunning_fog_index)
    # syntactic complexity
    data['SyntacticComplexity'] = data['text'].apply(syntactic_complexity)
    # average sentence length
    data['AverageSentenceLength'] = data['text'].apply(average_sentence_length)

    return data

def scale(x, scaling_algo):
    x_text = x['text']
    x = x.drop(['text'], axis=1)  # Drop 'text' column before scaling
    x_scaled = scaling_algo.transform(x)  # Use transform instead of fit_transform
    x = pd.DataFrame(x_scaled, columns=x.columns, index=x.index)  # Keep original index
    return pd.concat([x_text, x], axis=1)

def add_columns(x, vec_size):
  for i in range(vec_size):
    x[f'doc2vec_{i+1}'] = x['doc2vec'].apply(lambda v: v[i])
  return x

def predict_generated(text_input, threshold = 0.60):

  # convert to df pattern so "add_features" will deal with it
  df = pd.DataFrame({'text': [text_input], 'generated': [2]}) # 2 = 'unknown'

  # add the features from the text
  df = add_features(df)

  df.drop(['generated'], axis=1, inplace=True)

  with open('models/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)
  df = scale(df, scaler)

  # using L1 results of train
  selected_features = pd.Index(['text', 'punctuation%', 'tone', 'neg', 'pos', 'compound', 'Richness', '%_comma',
        '%_period', '%_q_mark', 'Nouns', 'Pronouns', 'Verbs', 'Adjectives',
        'Adverbs', 'Prepositions', 'Conjunctions', 'Interjections',
        'Entities_ratio', '%_stopwords', 'text_length', 'num_words',
        'FleschReadingEase', 'FleschKincaidGradeLevel', 'GunningFogIndex',
        'SyntacticComplexity', 'AverageSentenceLength'])

  df = df[selected_features]

  # Load the doc2vec model from the pickle file
  with open('models/doc2vec_model.pkl', 'rb') as file:
      doc2vec_model = pickle.load(file)

  # using doc2vec model from train
  df['doc2vec'] = [doc2vec_model.infer_vector(df['text'])]

  # convert the vector to features
  df = add_columns(df.copy(), doc2vec_model.vector_size)

  # drop vector + text
  df.drop(columns=['doc2vec', 'text'], axis=1, inplace=True)

  # Load the svm model from the pickle file
  with open('models/svm_model.pkl', 'rb') as file:
      svm_ = pickle.load(file)

  # apply ideal SVM with 80% thereshold for Human
  y_pred = (svm_.predict_proba(df)[:, 1] > threshold).astype(int)

  # print the result for the above text
  y_pred_map = np.where(y_pred == 1, "AI", "Human")
  print(f"\n{y_pred_map[0]} Generated")