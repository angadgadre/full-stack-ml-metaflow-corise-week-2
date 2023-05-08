# done: modify this custom model to your liking. Check out this tutorial for more on this class: https://outerbounds.com/docs/nlp-tutorial-L2/
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.layers import Dense, LeakyReLU

class NbowModel():
    def __init__(self, vocab_sz):

        self.vocab_sz = vocab_sz

        # Instantiate the CountVectorizer
        self.cv = CountVectorizer(
            min_df=.0025, max_df = .5, stop_words='english', # reduce min freq to 0.25%, and max freq to 50%.
            strip_accents='ascii', max_features=self.vocab_sz
        )

        # Define the keras model
        # boilerplate from week-2 below
        inputs = tf.keras.Input(shape=(self.vocab_sz,), 
                                name='input')
        x = layers.Dropout(0.10)(inputs)
        x = layers.Dense(
            15, activation="relu",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)
        )(x)
        predictions = layers.Dense(1, activation="sigmoid",)(x)
        self.model = tf.keras.Model(inputs, predictions)
        opt = optimizers.Adam(learning_rate=0.002)
        self.model.compile(loss="binary_crossentropy", 
                           optimizer=opt, metrics=["accuracy"])

        # week-2 mucking around with keras model
        # self.model = tf.keras.Sequential([
        #             layers.Dense(10, input_shape=(self.vocab_sz,), activation=None, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)),
        #             layers.LeakyReLU(alpha=0.1),
        #             Dense(1, activation='sigmoid')
        #         ])
        
        # week-2 compile the model
        # self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        

    def fit(self, X, y):
        print(X.shape)
        print(X[0])
        res = self.cv.fit_transform(X).toarray()
        self.model.fit(x=res, y=y, batch_size=32, 
                       epochs=10, validation_split=.2)
    
    def predict(self, X):
        print(X.shape)
        print(X[0])
        res = self.cv.transform(X).toarray()
        return self.model.predict(res)
    
    def eval_acc(self, X, labels, threshold=.5):
        return accuracy_score(labels, 
                              self.predict(X) > threshold)
    
    def eval_rocauc(self, X, labels):
        return roc_auc_score(labels,  self.predict(X))

    @property
    def model_dict(self): 
        return {'vectorizer':self.cv, 'model': self.model}

    @classmethod
    def from_dict(cls, model_dict):
        "Get Model from dictionary"
        nbow_model = cls(len(
            model_dict['vectorizer'].vocabulary_
        ))
        nbow_model.model = model_dict['model']
        nbow_model.cv = model_dict['vectorizer']
        return nbow_model
