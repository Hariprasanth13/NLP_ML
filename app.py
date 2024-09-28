import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

w2v_model = api.load("word2vec-google-news-300")

def sent_embedding(sentence):
    embeddings=[]
    for word in sentence.split():
        if word in w2v_model:
            embeddings.append(w2v_model[word])
    if embeddings:
        return np.mean(embeddings, axis = 0) # return the mean of the embeddings for words in a sentence.
    else:
        return np.zeros(w2v_model.vector_size) #return zero vector if the none of the words are found in the model
    
data = {
    'Feedback': [
        'Good product, works well.', 
        'Very dissatisfied with the service.', 
        'Delivery was fast!', 
        'The quality could be better.', 
        'Excellent experience, highly recommend it!',
        'I love it.',  # Short feedback
        'Horrible experience! Never buying again.',  # Longer negative feedback
        'It was okay, nothing special.',  # Neutral feedback
        'Customer support was prompt and resolved my issue quickly.',  # Longer positive feedback
        'Could use some improvement but overall decent value for the price.'
    ],
    'rating': [10, 2, 8, 4, 9, 9, 1, 5, 7, 6],
    'Category': ['A', 'B', 'A', 'C', 'B','A', 'B', 'A', 'C', 'B'],
    'Target': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]  # Example target (1=positive, 0=negative)
}

df = pd.DataFrame(data)

numerical_features = 'rating'
catergorical_feature = ['Category']

# applying standard scaler
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(df[[numerical_features]])

#Applying one hot encoding for categorical data
ohe = OneHotEncoder()
categorical_scaled = ohe.fit_transform(df[catergorical_feature]).toarray()

textual_data = df['Feedback'].apply(sent_embedding).tolist()
textual_data = np.array(textual_data)
print(numerical_scaled.shape)
print(categorical_scaled.shape)
print(textual_data.shape)

x = np.hstack([numerical_scaled,categorical_scaled,textual_data]) # All the features are converted to array before stacking
y = df['Target'].values

# Train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

rf = RandomForestClassifier()
rf.fit(x_train,y_train)
pred = rf.predict(x_test)

print("Accuracy Score: ",accuracy_score(y_test,pred))



