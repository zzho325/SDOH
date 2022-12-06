from pathlib import Path
import os, sys, string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import matplotlib.pyplot as plt

BASE_DIR = Path(os.path.dirname(os.path.abspath(sys.argv[0])))

files = os.listdir('Annotation_pt')
print(f'processing file {files[0]}')
sdh_notes = pd.read_excel(f'Annotation_pt/{files[0]}', engine='openpyxl')
for file in files[1:]:
  print(f'processing file {file}')
  sdh_notes = pd.concat([sdh_notes, pd.read_excel(f'Annotation_pt/{file}', engine='openpyxl')], axis=0, sort=False)

print('Dataset shape', sdh_notes.shape)

sdh_notes.rename(columns={ \
  'Income and other general financial insecurity': 'income',
  'Housing Insecurity': 'housing',
  'Insurance insecurity': 'insurance',
  'Alcohol Use': 'alcohol',
  'Substance Use': 'substance',
  'Tobacco Use': 'tobacco',
  'Gender Identity / Sexual Orientation' : 'gender',
  'Disability': 'disability'
}, inplace=True)

class_names = ['income', 'housing', 'insurance', 'alcohol', 'substance', 'tobacco', 'disability']

for c in class_names:
  length = sdh_notes[sdh_notes[c] == 1].shape[0]
  print(f'class {c} : {length}')

tmp = sdh_notes.groupby('patient_id').size().reset_index(name='counts')
print(f'Number of patients {tmp.shape[0]}')

train_pt, test_pt = train_test_split(tmp, test_size=0.1, shuffle=True)
train_pt, val_pt = train_test_split(train_pt, test_size=0.11, shuffle=True)
train = pd.merge(sdh_notes, train_pt, how="inner", on=["patient_id"], 
                 suffixes=("_x", "_y"), copy=True, indicator=True, validate=None,)
val = pd.merge(sdh_notes, val_pt, how="inner", on=["patient_id"], 
               suffixes=("_x", "_y"), copy=True, indicator=True, validate=None,)
test = pd.merge( sdh_notes, test_pt, how="inner", on=["patient_id"], 
                suffixes=("_x", "_y"), copy=True, indicator=True, validate=None,)
train.drop(["patient_id", "counts", "_merge"], axis=1, inplace=True)
val.drop(["patient_id", "counts", "_merge"], axis=1, inplace=True)
test.drop(["patient_id", "counts", "_merge"], axis=1, inplace=True)

test.to_csv(BASE_DIR / 'test.csv', index=False)
val.to_csv(BASE_DIR / 'val.csv', index=False)
train.to_csv(BASE_DIR / 'train.csv', index=False)

print('Records in training set:',len(train))
print('Records in validation set:',len(val))
print('Records in test set:',len(test))

# this function receives comments and returns clean word-list
def clean_doc(text_record):
    # split tokens by white space
    tokens = text_record.split()
    # remove punctuation from each string
    table = str.maketrans({key: None for key in string.punctuation})
    tokens = [token.translate(table) for token in tokens]
    # remove tokens that are not alphabetic
    tokens = [token for token in tokens if token.isalpha()]
    # convert letters to lower case
    
    tokens = [token.lower() for token in tokens]
    #stem tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token.lower()) for token in tokens]
    # tokens = stemmer.stem(tokens.lower())
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    
    tokens = [token for token in tokens if token not in stop_words]
    # remove short words (one letter)
    tokens = [token for token in tokens if len(token) > 1]
    # lemmatization
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(token,'v') for token in tokens]
    sentence = ' '.join(tokens)
    return sentence
  
# preprocess all comment texts in the trainning set and testing set
train_text_clean = [clean_doc(comment) for comment in train.RESULT_TVAL]
val_text_clean = [clean_doc(comment) for comment in val.RESULT_TVAL]
test_text_clean = [clean_doc(comment) for comment in test.RESULT_TVAL]

Y_train = train[class_names]
Y_val = val[class_names]
Y_test = test[class_names]

print('Data cleaned')

tf.get_logger().setLevel('ERROR')

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_model = hub.KerasLayer(tfhub_handle_encoder)

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['sequence_output']
  net = layers.Bidirectional(layers.LSTM(128, dropout=0.2))(net)
  net = layers.Dense(len(class_names), activation='sigmoid', name='classifier')(net)
  return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()
classifier_model.summary()

optimizer = Adam(
    learning_rate=5e-05, # HF recommendation
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0
)
loss = tf.keras.losses.BinaryCrossentropy()
metrics = [tf.metrics.BinaryAccuracy(), tf.metrics.AUC()]
classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=BASE_DIR / "fine_tuning_bert_bilstm.keras",
        save_best_only=True,
        monitor="val_loss")
]

print(f'Training model with {tfhub_handle_encoder}')
history = classifier_model.fit(x=np.array(train_text_clean), y=Y_train.to_numpy(), 
                               validation_data=(pd.Series(val_text_clean), Y_val), 
                               epochs=30, callbacks=callbacks, verbose=2)

test_loss, test_acc, test_auc = classifier_model.evaluate(pd.Series(test_text_clean), Y_test, verbose=2)
print(f"Test accuracy: {test_acc:.3f}, Test AUC: {test_auc:.3f}")

accuracy = history.history["binary_accuracy"]
val_accuracy = history.history["val_binary_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig(BASE_DIR / 'model_accuracy.png')

plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
# plt.show()
plt.savefig(BASE_DIR / 'model_loss.png')