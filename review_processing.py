
# coding: utf-8

# In[25]:


#import the required libraries

import pickle
from nltk.corpus import wordnet as wn
import pandas as pd
import json
import difflib
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# In[3]:


#level 1 food items from Foursquare post clustering

cluster_level_1 = set(pickle.load( open( "cluster_level_1.pickle", "rb" ) ))


# In[41]:


print('no of level 1 items post clustering = ', len(cluster_level_1))


# In[5]:


#level 2 food items from Foursquare post clustering

cluster_level_2 = set(pickle.load( open( "cluster_level_2.pickle", "rb" ) ))


# In[42]:


print('no of level 2 items post clustering = ',len(cluster_level_2))


# In[8]:


#level 1 food items from foursquare pre clustering

level1 = set(pickle.load( open( "level1.pickle", "rb" ) ))


# In[43]:


print('no of level 1 items post clustering = ', len(level1))


# In[44]:


#level 2 food items from foursquare post clustering

level2 = set(pickle.load( open( "level2.pickle", "rb" ) ))


# In[45]:


print('no of level 2 items post clustering = ', len(level2))


# In[10]:


fs = level1.union(level2)


# In[12]:


print(wn.synsets('food')[0].definition())
print(wn.synsets('food')[2].definition())
print(wn.synsets('food')[2].definition())


# In[46]:


food = wn.synset('food.n.02')
nltk_corpus = set([w for s in food.closure(lambda s:s.hyponyms()) for w in s.lemma_names()])

print('size of NLTK corpus = ',len(nltk_corpus))


# In[15]:


#replacing '_' with ' '  and also adding splitted words to create a modified corpus

nltk_corpus_modified = set()

for word in nltk_corpus:
    nltk_corpus_modified.add(word.replace('_',' '))
    splitted_string = word.split('_')
    if len(splitted_string)>1:
        for subword in splitted_string:
            nltk_corpus_modified.add(subword)


# In[47]:


print('size of NLTK corpus modified = ',len(nltk_corpus_modified))


# In[18]:


#import the google dataset of 20k most common words in English

common_words = set(line.strip() for line in open('20k.txt'))


# In[49]:


print(len(common_words))


# In[33]:


#removing the food words from the set of common words


# In[52]:


common_words_wo_food = common_words - nltk_corpus_modified


# In[53]:


print(len(common_words_wo_food))


# In[54]:


l = set(common_words.intersection(nltk_corpus_modified))


# In[55]:


food_added = l - set(['farm',
 'seafood',
 'crab',
 'strawberry',
 'wheat',
 'baked',
 'spaghetti',
 'garlic',
 'jelly',
 'creme',
 'greens',
 'pea',
 'cucumber',
 'turtle',
 'chop',
 'german',
 'cracker',
 'bone',
 'dove',
 'barbados',
 'tea',
 'goose',
 'fry',
 'kiwi',
 'rum',
 'honey',
 'root',
 'chocolate',
 'cookie',
 'butter',
 'rye',
 'soybean',
 'cake',
 'cheese',
 'shrimp',
 'ribs',
 'corn',
 'sausage',
 'spice',
 'lime',
 'cherry',
 'berry',
 'blackberry',
 'potatoes',
 'apple',
 'ginger',
 'potato',
 'lobster',
 'onion',
 'tomato',
 'leaf',
 'steak',
 'pasta',
 'rib',
 'coffee',
 'toast',
 'fries',
 'salmon',
 'mustard',
 'spinach',
 'broccoli',
 'nan',
 'raspberry',
 'grape',
 'cinnamon',
 'mandarin',
 'turkey',
 'rabbit',
 'mango',
 'chicken',
 'stew',
 'cottage',
 'beef',
 'crust',
 'cereal',
 'ham',
 'patty',
 'smoked',
 'tuna',
 'bean',
 'egg',
 'lemon',
 'soda',
 'saltwater',
 'banana',
 'peach',
 'belly',
 'fig',
 'cranberry',
 'cocoa',
 'lima',
 'pie',
 'liver',
 'plum',
 'pigeon',
 'bologna',
 'swedish',
 'pastry',
 'pig',
 'pumpkin',
 'nut',
 'guinea',
 'cod',
 'orange',
 'buffalo',
 'chili',
 'shellfish',
 'goa',
 'almond',
 'pork',
 'mushroom',
 'bacon',
 'lettuce',
 'roast',
 'chips',
 'meat',
 'coconut',
 'milk',
 'cabbage',
 'pineapple',
 'yogurt',
 'carrot',
 'coco',
 'fish',
 'oyster',
 'pepper',
 'rice'])


# In[56]:


common_words_wo_food = common_words_wo_food.union(food_added)


# In[57]:


#loading the reviwes data

file = open('dataset/review.json','r')


# In[58]:


def get_food_items(review):
    vect = CountVectorizer(stop_words = ENGLISH_STOP_WORDS)
    vect.fit([review])
    l = list(vect.get_feature_names())
    nouns = list(TextBlob(review).noun_phrases)
    
    for word in l:
        flag = 0
        for n in nouns:
            if word in n.split():
                flag = 1    
        if flag ==0:
            nouns.append(word)
            
    nouns_mod = []

    for item in nouns:
        if item not in common_words_wo_food and not item.isdigit():
            nouns_mod.append(item)
        
    return nouns_mod


# In[59]:


d = json.loads(next(file))
review = d['text']
print(review)
food_items = get_food_items(review)

for item in food_items:  
    #print(item,process.extract(item, level1, limit=3),process.extract('item', level2, limit=1))
    print(item,difflib.get_close_matches(item,level1,n=3,cutoff=0.7),difflib.get_close_matches(item,level2,n=1,cutoff=0.7))

