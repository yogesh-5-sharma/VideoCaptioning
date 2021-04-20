import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import json

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, batch_size, features, video_ids, cap_sequences, vocab_size, max_length):
    self.batch_size = batch_size
    self.features = features
    self.video_ids = video_ids
    self.cap_sequences = cap_sequences
    self.vocab_size = vocab_size
    self.max_len = max_length
    self.on_epoch_end()

  def on_epoch_end(self):
    self.indices = np.arange(len(self.cap_sequences))
    np.random.shuffle(self.indices)

  def __len__(self):
    return len(self.indices)//self.batch_size
  
  def __getitem__(self, index):
    idx_range = self.indices[index*self.batch_size : (index+1)*self.batch_size]
    return self.__data_generation(idx_range)
  
  def __data_generation(self, list_index):
    X1=[]
    X2=[]
    Y=[]
    for i in list_index:
      id=self.video_ids[i]
      X1.append(self.features[id])
      X2.append(tf.keras.utils.to_categorical(self.cap_sequences[i][:-1],num_classes= self.vocab_size))
      Y.append(tf.keras.utils.to_categorical(self.cap_sequences[i][1:],num_classes= self.vocab_size))
    X1=np.array(X1)
    X2=tf.keras.preprocessing.sequence.pad_sequences(X2, maxlen= self.max_len, padding='post')
    Y=tf.keras.preprocessing.sequence.pad_sequences(Y, maxlen= self.max_len, padding='post')
    return [X1,X2], Y



class Video_Captioning_Model():
  def __init__(self, ):
    
    training_features_directory = 'dataset\yt_allframes_vgg_fc7_train.txt'
    training_sents_directory = 'dataset\sents_train_lc_nopunc.txt'
    validation_features_directory = 'dataset\yt_allframes_vgg_fc7_val.txt'
    validation_sents_directory = 'dataset\sents_val_lc_nopunc.txt'
    
    print("Loading Dataset...")
    raw_training_features = self.__getFeatures(training_features_directory)
    raw_training_sents = self.__getSents(training_sents_directory)
    raw_validation_features = self.__getFeatures(validation_features_directory)
    raw_validation_sents = self.__getSents(validation_sents_directory)
    
    self.frames_limit = 80
    self.vocab_size = 5000
    
    self.saved_model_directory = 'saved_models'
    
    self.training_features, self.training_ids, self.training_captions = self.preprocess_data(raw_training_features,
                                                                                             raw_training_sents)
    self.validation_features, self.validation_ids, self.validation_captions = self.preprocess_data(raw_validation_features, 
                                                                                                   raw_validation_sents)
    
    self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words= self.vocab_size,
                                                          filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
                                                          oov_token= "<oov>")
    self.tokenizer.fit_on_texts(self.training_captions + self.validation_captions)
    self.training_seq = self.tokenizer.texts_to_sequences(self.training_captions)
    self.validation_seq = self.tokenizer.texts_to_sequences(self.validation_captions)
    self.max_length = max([len(x) for x in self.training_seq])
    
    tokenizer_json = self.tokenizer.to_json()
    with open(os.path.join(self.saved_model_directory, 'tokenizer.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    
    self.num_timesteps_encoder = 80
    self.num_tokens_encoder = 4096
    self.latent_dims = 1000
    self.num_timesteps_decoder = self.max_length - 1
    self.num_tokens_decoder = self.vocab_size
    
    return
  
  def train(self):
    self.model = self.build_model()
    self.model.compile(
      optimizer= 'adam',
      loss= 'categorical_crossentropy',
      metrics= ['accuracy']
    )
    
    print("Creating Generators...")
    training_generator = DataGenerator(
        batch_size= 32,
        features= self.training_features,
        video_ids= self.training_ids,
        cap_sequences= self.training_seq,
        max_length= self.max_length-1,
        vocab_size= self.vocab_size
    )

    validation_generator = DataGenerator(
      batch_size= 32,
      features= self.validation_features,
      video_ids= self.validation_ids,
      cap_sequences= self.validation_seq,
      max_length= self.max_length-1,
      vocab_size= self.vocab_size
    )
    
    print("Starting Training...")
    try:
      self.history = self.model.fit(
        training_generator,
        epochs= 500,
        validation_data = validation_generator
      )
    except KeyboardInterrupt:
      print("Keyboard interrupt!!")
    
    print("Saving Models...")
    self.model.save(os.path.join(self.saved_model_directory, 'model.h5'))

    self.encoder_model = tf.keras.models.Model(self.encoder_input, self.encoder_state)
    decoder_state_input_h = tf.keras.layers.Input(shape= (self.latent_dims,))
    decoder_state_input_c = tf.keras.layers.Input(shape= (self.latent_dims,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_output_lstm, state_h, state_c = self.decoder_lstm(self.decoder_input, initial_state= decoder_state_inputs)
    decoder_state_outputs = [state_h, state_c]
    decoder_output_dense = self.decoder_dense(decoder_output_lstm)
    self.decoder_model = tf.keras.models.Model(inputs=[self.decoder_input, decoder_state_inputs], outputs=[decoder_output_dense,decoder_state_outputs])

    self.encoder_model.save(os.path.join(self.saved_model_directory, 'encoder_model.h5'))
    self.decoder_model.save_weights(os.path.join(self.saved_model_directory, 'decoder_model_weights.h5'))
    np.save(os.path.join(self.saved_model_directory, 'history.npy'), self.history.history)
    print("Done")

  def retrain(self):
    self.model = tf.keras.models.load_model(os.path.join(self.saved_model_directory, 'model.h5'))
    
    print("Creating Generators...")
    training_generator = DataGenerator(
        batch_size= 32,
        features= self.training_features,
        video_ids= self.training_ids,
        cap_sequences= self.training_seq,
        max_length= self.max_length-1,
        vocab_size= self.vocab_size
    )

    validation_generator = DataGenerator(
      batch_size= 32,
      features= self.validation_features,
      video_ids= self.validation_ids,
      cap_sequences= self.validation_seq,
      max_length= self.max_length-1,
      vocab_size= self.vocab_size
    )
    
    print("Re-Starting Training...")
    try:
      self.history = self.model.fit(
        training_generator,
        epochs= 1,
        validation_data = validation_generator
      )
    except KeyboardInterrupt:
      print("Keyboard interrupt!!")
    
    print("Saving Models...")
    self.model.save(os.path.join(self.saved_model_directory, 'model.h5'))

    self.encoder_model = tf.keras.models.Model(self.encoder_input, self.encoder_state)
    decoder_state_input_h = tf.keras.layers.Input(shape= (self.latent_dims,))
    decoder_state_input_c = tf.keras.layers.Input(shape= (self.latent_dims,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_output_lstm, state_h, state_c = self.decoder_lstm(self.decoder_input, initial_state= decoder_state_inputs)
    decoder_state_outputs = [state_h, state_c]
    decoder_output_dense = self.decoder_dense(decoder_output_lstm)
    self.decoder_model = tf.keras.models.Model(inputs=[self.decoder_input, decoder_state_inputs], outputs=[decoder_output_dense,decoder_state_outputs])

    self.encoder_model.save(os.path.join(self.saved_model_directory, 'encoder_model.h5'))
    self.decoder_model.save_weights(os.path.join(self.saved_model_directory, 'decoder_model_weights.h5'))
    np.save(os.path.join(self.saved_model_directory, 'history1.npy'), self.history.history)
    print("Done")
  
  def build_model(self):
    print("Building Model...")
    
    self.encoder_input = tf.keras.layers.Input(shape = (self.num_timesteps_encoder, self.num_tokens_encoder), name= 'encoder_inputs')
    self.encoder_lstm = tf.keras.layers.LSTM(units= self.latent_dims, return_state= True, return_sequences= True, name= 'encoder_lstm')
    _, state_h, state_c = self.encoder_lstm(self.encoder_input)
    self.encoder_state = [state_h, state_c]
    
    self.decoder_input = tf.keras.layers.Input(shape = (self.num_timesteps_decoder, self.num_tokens_decoder), name= 'decoder_inputs')
    self.decoder_lstm = tf.keras.layers.LSTM(units= self.latent_dims, return_state= True, return_sequences= True, name= 'decoder_lstm')
    self.decoder_output, _, _ = self.decoder_lstm(self.decoder_input, initial_state= self.encoder_state)
    self.decoder_dense = tf.keras.layers.Dense(units= self.vocab_size, activation= 'softmax', name= 'decoder_dense')
    self.decoder_output = self.decoder_dense(self.decoder_output)
    
    model = tf.keras.models.Model([self.encoder_input, self.decoder_input], self.decoder_output)
    return model
  
  def preprocess_data(self, features, sents):
    result_features = {}
    result_ids = []
    result_captions = []
    
    for x,y in features.items():
      video_features = None
      if y.shape[0] > self.frames_limit:
        idx = np.linspace(0, y.shape[0]-1, self.frames_limit).astype('int')
        video_features = y[idx]
      else:
        video_features = np.pad(y, ((0, self.frames_limit - y.shape[0]), (0, 0)))
      result_features[x]=video_features
    
    for id, captions in sents.items():
      for caption in captions:
        result_ids.append(id)
        cap = "<bos> " + caption + " <eos>"
        result_captions.append(cap)
    
    return result_features, result_ids, result_captions
  
  def __getFeatures(self, directory):
    with open(directory, "r") as f:
      data = f.read()

    datalist = data.split()
    features = {}

    for x in datalist:
      row = x.split(',')
      id = row[0].split('_')[0]
      if id not in features:
        features[id]=[]
      features[id].append(np.asarray(row[1:], dtype=np.float))

    for x in features:
      features[x] = np.array(features[x])

    return features


  def __getSents(self, directory):
    with open(directory, "r") as f:
      data = f.read()

    datalist = data.split('\n')
    sents = {}

    for x in datalist:
      row = x.split('\t')
      if len(row)<2:
        continue
      id = row[0]
      if id not in sents:
        sents[id] = []
      sents[id].append(row[1])

    return sents