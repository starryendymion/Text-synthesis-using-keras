from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import numpy as np
import pickle


absolute_path = os.path.dirname(__file__)

relative_path_model = "textsyn_model_new.h5"
full_path_model = os.path.join(absolute_path, relative_path_model)

relative_path_tokenizer= "mysimpletokenizer"
full_path_tokenizer = os.path.join(absolute_path, relative_path_tokenizer)


def generate_text(model,tokenizer):
 seed_text=str(input("Enter the first few words [ max = 10 ] : "))
 n=5
 seed_text = seed_text.split(' ') 

 tokenizer.fit_on_texts(seed_text)
 sequences=tokenizer.texts_to_sequences(seed_text)
 sequences = [s for s in sequences if len(s) > 0]  # remove empty sequences

 feature=pad_sequences([sequences], padding='post', maxlen=10)
 feature=feature.reshape(-1,)

 generated_text=np.array([])
 for i in range(n):
    x_pred=np.reshape(feature,(1,10))
    preds=model.predict(x_pred,verbose=0)[0]
    next_word_index=np.argmax(preds)
    next_word=tokenizer.index_word[next_word_index]
    generated_text=np.concatenate([generated_text,[next_word]])
    feature=np.concatenate([feature[1:],[next_word_index]],axis=0)

    
 
 print('Next few words: ',' '.join(generated_text))



mymodel     =  keras.models.load_model(full_path_model)
mytokenizer = pickle.load(open(full_path_tokenizer, 'rb'))

while True:
  print("------------------------------------------------------------------------------------------------------------------------ ")  
  ans=str(input("Make a prediction? Y/N : "))
  if ans.lower()=='y':
    generate_text(model=mymodel,tokenizer=mytokenizer)
  elif ans.lower()=='n':
     break
  else:
    print("Please enter either Y or N!")
    continue
  print("------------------------------------------------------------------------------------------------------------------------ ")  
    
  
