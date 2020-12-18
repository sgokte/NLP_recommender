
import re
import numpy as np
import pandas as pd
import pickle

import streamlit as st
from io import BytesIO, StringIO
from PIL import Image
import requests
from io import BytesIO

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


# spacy for lemmatization
import spacy
from nltk.stem.wordnet import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[2]:


metadata = pd.read_csv("metadata_final.csv", encoding='latin-1')
metadata.head()


# # **Load Saved Models & Objects**

# In[134]:


# Load Doc2Vec Model
model_d2v = Doc2Vec.load('optimal_d2v_model.doc2vec')


# In[6]:


# TaggedDocument Iterartor for saved reviews corpus

class TaggedDocumentIterator(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc.split(), tags=[self.labels_list[idx]])


# In[7]:


# Load Vectorizer
with open('tagged_doc_corpus.pkl', 'rb') as f:
  tagged_doc_corpus = pickle.load(f)

# Load LDA Model
with open('optimal_LDA_Model.pkl', 'rb') as f:
  lda_model = pickle.load(f)

# Load Vectorizer
with open('optimal_vectorizer.pkl', 'rb') as f:
  vectorizer = pickle.load(f)


# # **Input Description Preprocessing**

# In[9]:


# NLTK Stop words
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


# In[10]:


# Function to remove bad/unwanted characters and converting to lowercase
def default_clean(text):
    '''
    Removes default bad characters
    '''
    if not (pd.isnull(text)):
    # text = filter(lambda x: x in string.printable, text)
      bad_chars = set(["@", "+", '/', "'", '"', '\\','(',')', '\\n', '?', '#', ',','.', '[',']', '%', '$', '&', ';', '!', ':',"*", "_", "=", "}", "{"])
    for char in bad_chars:
        text = text.replace(char, " ")
    text = re.sub('\d+', "", text)
    return text.lower()

# Function for sentences to words
def sent_to_words(sentences):
    yield(gensim.utils.simple_preprocess(str(sentences), deacc=True))  # deacc=True removes punctuations

# Function for stopwords
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Function for bigrams and trigrams
def bigram_trigrams(texts):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(texts, min_count=2, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# In[42]:


def lemmatization(text, lemmer = WordNetLemmatizer()):
    '''
    Removes stopwords and does lemmatization
    '''
    text_out = []
    for word_list in text:
      text_lemmatized = []
      for word in word_list:
        if '_' not in word and len(word) > 3:
          text_lemmatized.append(lemmer.lemmatize(word))
        elif '_' in word:
          #ngram_word = word.replace('_', ' ')
          text_lemmatized.append(word)
      text_out.append(' '.join(text_lemmatized))
    
    return text_out


# In[45]:


# Define function to predict topic for a given text document.
#nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def preprocess_description(text):
    global sent_to_words
    global lemmatization

    # Clean with simple_preprocess
    mytext = default_clean(text)
    mytext_2 = list(sent_to_words(mytext))

    # Remove Stop Words
    mytext_3 = remove_stopwords(mytext_2)

    # Form Bigrams
    mytext_4 = bigram_trigrams(mytext_3)

    # Step 2: Lemmatize
    #mytext_5 = lemmatization(mytext_4, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    mytext_5 = lemmatization(mytext_4)

    return mytext_5


# In[14]:


# Function to generate topic keywords and weights
def topic_word_matrix(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    keywords_weights = []
    normalized_weights = []
    weight_total = lda_model.components_.sum(axis=1)
    for i, weight in enumerate(weight_total):
      normalized_weights.append(lda_model.components_[i] / weight)

    for topic_weights in normalized_weights:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
        keywords_weights.append(topic_weights.take(top_keyword_locs))
    return topic_keywords,keywords_weights


# Function to predict topic distribution for new location
def predict_topic(vectorizer, lda_model, clean_text):

  topic_keywords, topic_keywords_weights = topic_word_matrix(vectorizer=vectorizer, 
                                                             lda_model=lda_model, 
                                                             n_words=20) 
  # Topic - Keywords Dataframe
  df_topic_keywords = pd.DataFrame(topic_keywords)
  df_topic_keywords.columns = ['Word-'+str(i) for i in range(df_topic_keywords.shape[1])]
  
  df_topic_keywords.index = ['Monuments/Historical Architectures/Spiritual Attractions', 'Mountains/Landscapes/Waterfalls', 'Beach/Seashores', 
                             'Temples/Church/Worship', 'Wildlife/Forests/National Parks', 'Gardens/City Parks', 
                             'Palace/Forts/Ancient Buildings', 'Museums/Indian Culture & History', 'Riverbank/Pilgrimage']
  
  # Vectorize transform cleaned text
  mytext = vectorizer.transform(clean_text)

  # LDA Transform
  topic_probability_scores = lda_model.transform(mytext)
  topic_words = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :].values.tolist()
  #top_topics_loc = np.where(topic_probability_scores > 0.1)[1]
  top_prob_scores = topic_probability_scores[0][np.argmax(topic_probability_scores)]
  #top_topic_words = df_topic_keywords.iloc[top_topics_loc, :].values.tolist()
  topics = df_topic_keywords.index[np.argmax(topic_probability_scores)]
  #top_topics = df_topic_keywords.index[top_topics_loc].values.tolist()
  return topics, topic_words, top_prob_scores


# # **Topic Inference and Recommendation Section**


input_location = 'Cherai Beach'


def get_recommendation(input_text,vectorizer,lda_model,doc2vec_model):
  clean_text = preprocess_description(input_text)
  sim_score = []
  # For tourist locations topic inference
  predicted_topic, topic_words, prob_score = predict_topic(vectorizer,lda_model,clean_text)

  # For tourist locations recommendation
  sample_df = metadata.loc[metadata['dominant_topic'] == predicted_topic].reset_index(drop=True)
  sentences = TaggedDocument(words = clean_text[0].split(), tags = ['new_location'])
  doc_tags = metadata['Place Name'].values.tolist()
  for loc in sample_df['Place Name']:
      idx = doc_tags.index(loc)
      score = doc2vec_model.docvecs.similarity_unseen_docs(doc2vec_model,sentences.words,list(tagged_doc_corpus)[idx].words)
      sim_score.append((loc,score))

  recommendation_list = sorted(sim_score, key = lambda x: x[1], reverse=True)[:10]
  locations = [loc[0] for loc in recommendation_list]
  similarity_score = [loc[1] for loc in recommendation_list]
  recommendation_df = metadata.loc[metadata['Place Name'].isin(locations)].reset_index(drop=True)
  recommendation_df['similarity_score'] = similarity_score

  return predicted_topic, prob_score, recommendation_df




# STREAMLIT code:

def main():
    
    # STREAMLIT code:
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background: url(https://i.postimg.cc/ZnHTP71s/aircraft-airplane-boat-1575833.jpg)"  class="page-holder bg-cover"> 
    <h1 style ="color:white;text-align:center;">TOURISTY</h1>
    <p style ="color:white;text-align:center;" class="text-white lead mb-5">A Novel Unsupervised NLP based Web application for Similar tourist locations recommendation</p>
    </header>
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)
    st.markdown("## --------------------------------------------------------------")
    #st.title("\t \t Touristy")
    st.markdown("#### 1. Enter detailed description of a location that you have visited in the past")
    st.markdown("#### 2. Use the slider to get recommendations as per your need!")
    st.markdown("## --------------------------------------------------------------")
    st.write("\n")

    description = st.text_area("Enter the Description","Type Here...")


    topic, prob_scores, recommendation_df = get_recommendation(description,vectorizer,lda_model,model_d2v)

    st.markdown("#### Choose the number of Locations")
    slider = st.slider("",1,5)
    st.info("#### NOTE: You can only get 5 locations reommended at a time")

    if st.button("Search"):
      if not description:
        st.warning("Please enter some discription")
        pass

      else:
        progress_bar = st.progress(0)
        succes_text = st.empty()

        count=100/slider
        num=count
        for i in range(slider):

          resp = requests.get(recommendation_df["img_source"][i])
          imgg = Image.open(BytesIO(resp.content))
          st.write("### "+ str(i+1)+". \t"+recommendation_df["Place Name"][i])
          st.write("( **State:** "+recommendation_df["State"][i]+", **City:** "+recommendation_df["City"][i]+")")
          link = '[[Click Here]]'+"("+recommendation_df["Link"][i]+")"
          st.markdown(link)


          st.image(imgg, width=500 , height=150, caption="Dominant Topic: "+str(recommendation_df["dominant_topic"][i])
            +" , Similarity Score: "+str(recommendation_df["similarity_score"][i]))

          progress_bar.progress(round(num))
          num=count+num

        succes_text.success("SUCCESS: These are the best locations for you")

        st.balloons()


if __name__ == '__main__':
    main()




