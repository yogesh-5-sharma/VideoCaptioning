import tensorflow as tf
import numpy as np
import os
import json
import nltk
import math
import heapq
import cv2

from cnn import CNN

class Video_Caption_Generator():
  def __init__(self):
    
    self.saved_model_directory = 'saved_models'
    if not os.path.exists(os.path.join(self.saved_model_directory, 'encoder_model.h5')) or \
       not os.path.exists(os.path.join(self.saved_model_directory, 'decoder_model_weights.h5')):
      raise Exception("No trained models found. Check for correct model filenames if already trained, else train model first.")
    
    if not os.path.exists(os.path.join(self.saved_model_directory, 'tokenizer.json')):
      raise Exception("Tokenizer object not found.")
    
    self.num_tokens_decoder = 5000
    self.latent_dims = 1000
    self.frames_limit = 80
    
    with open(os.path.join(self.saved_model_directory, 'tokenizer.json')) as fp:
      tokenizer_json = json.load(fp)
    
    self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    self.encoder_model, self.decoder_model = self.__get_inference_model()
    self.cnn_model = CNN()
    
    self.caption_limit = 30
    
    self.testing_features_directory = 'dataset\yt_allframes_vgg_fc7_test.txt'
    self.testing_sents_directory = 'dataset\sents_test_lc_nopunc.txt'
    
  def __get_inference_model(self):
    e_model = tf.keras.models.load_model(os.path.join(self.saved_model_directory, 'encoder_model.h5'))

    d_input = tf.keras.layers.Input(shape= (None, self.num_tokens_decoder))
    d_input_h = tf.keras.layers.Input(shape= (self.latent_dims,))
    d_input_c = tf.keras.layers.Input(shape= (self.latent_dims,))
    d_input_state = [d_input_h, d_input_c]
    d_lstm = tf.keras.layers.LSTM(self.latent_dims, return_state= True, return_sequences= True)
    d_dense = tf.keras.layers.Dense(units= self.num_tokens_decoder, activation= 'softmax')
    
    d_output, d_state_h, d_state_c = d_lstm(d_input, initial_state= d_input_state)
    d_output_state = [d_state_h, d_state_c]
    d_output = d_dense(d_output)
    d_model = tf.keras.models.Model(inputs= [d_input, d_input_state], outputs= [d_output, d_output_state])
    d_model.load_weights(os.path.join(self.saved_model_directory, 'decoder_model_weights.h5'))

    return e_model, d_model
  
  def __convert_sequence_to_text(self, candidate_captions, beam_width):
    candidate_captions = list(map(lambda x: (x[0],x[1]), candidate_captions))
    candidate_captions = sorted(candidate_captions, key= lambda x: x[0], reverse= True)
    resulting_captions = []
    resulting_probabilities = []
    for candidate in candidate_captions:
      sequence= candidate[1]
      sequence= sequence[1:]
      if self.tokenizer.index_word[sequence[-1]] == "<eos>":
        sequence = sequence[:-1]
      if self.tokenizer.index_word[sequence[-1]] == "a" and beam_width <=5:
        sequence = sequence[:-1]
      caption = self.tokenizer.sequences_to_texts([sequence])[0]
      resulting_captions.append(caption)
      resulting_probabilities.append(candidate[0])
    
    return resulting_captions, resulting_probabilities
  
  def __generate_captions(self, video_features, beam_width):
    video_features = np.expand_dims(video_features, axis=0)
    state_values = self.encoder_model.predict(video_features)
    candidate_captions = [[0.0, [self.tokenizer.word_index['<bos>']], state_values]]
    
    for i in range(self.caption_limit):
      new_candidates = []
      all_done = 1
      for candidate in candidate_captions:
        prob = candidate[0]
        sequence = candidate[1]
        state = candidate[2]
        if self.tokenizer.index_word[sequence[-1]] == "<eos>":
          if len(new_candidates) < beam_width:
            heapq.heappush(new_candidates, (prob, sequence, state))
          elif prob > new_candidates[0][0]:
            heapq.heappushpop(new_candidates, (prob, sequence, state))
          continue
        all_done = 0
        decoder_input_data = np.zeros(shape = (1, 1, self.num_tokens_decoder))
        decoder_input_data[0, 0, sequence[-1]] = 1
        probabilities, state_output = self.decoder_model.predict([decoder_input_data, state])
        for j in range(1, self.num_tokens_decoder):
          if probabilities[0, 0, j] > 0.0 and j != sequence[-1]:
            new_prob = (prob*len(sequence) + math.log(probabilities[0, 0, j]))/(1+len(sequence))
            new_sequence = sequence.copy()
            new_sequence.append(j)
            if len(new_candidates) < beam_width:
              heapq.heappush(new_candidates, (new_prob, new_sequence, state_output))
            elif new_prob > new_candidates[0][0]:
              heapq.heappushpop(new_candidates, (new_prob, new_sequence, state_output))
      
      if all_done == 1:
        break
      candidate_captions = new_candidates.copy()
    
    return self.__convert_sequence_to_text(candidate_captions, beam_width)
  
  def get_video_captions(self, video_directory, beam_width=5):
    
    if not os.path.exists(video_directory):
      raise Exception("Invalid Video directory")
    
    print("Reading Video Frames...")
    cap = cv2.VideoCapture(video_directory)
    
    frames = []
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break
      frames.append(frame)
    cap.release()
    
    frames = np.array(frames)
    if frames.shape[0] > self.frames_limit:
      frames_idx = np.linspace(0, frames.shape[0]-1, self.frames_limit).astype('int')
      frames = frames[frames_idx]
    
    print("Extracting Features...")
    features = self.cnn_model.extract_features(frames)
    features = np.pad(features, ((0,self.frames_limit - features.shape[0]), (0,0)))
    
    print("Generating Captions...")
    captions, probabilities = self.__generate_captions(features, beam_width)
    
    return captions
  
  def get_bleu_score(self, references, hypothesis):
    references = list(map(lambda x: x.split(), references))
    hypothesis = list(map(lambda x: x.split(), hypothesis))
    bleu_scores = []
    for hypo in hypothesis:
      score = nltk.translate.bleu_score.sentence_bleu(references, hypo)
      bleu_scores.append(score)
    return max(bleu_scores)
  
  def get_meteor_score(self, references, hypothesis):
    try:
      nltk.data.find('corpora\wordnet')
    except LookupError:
      nltk.download('wordnet')
      
    meteor_scores = []
    for hypo in hypothesis:
      score = nltk.translate.meteor_score.meteor_score(references, hypo)
      meteor_scores.append(score)
    return max(meteor_scores)
  
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
  
  def test(self, beam_width_values):
    testing_features = self.__getFeatures(self.testing_features_directory)
    testing_sents = self.__getSents(self.testing_sents_directory)
    for key in testing_features:
      vid_features = testing_features[key]
      if vid_features.shape[0] < self.frames_limit:
        vid_features = np.pad(vid_features, ((0, self.frames_limit-vid_features.shape[0]), (0,0)))
      else:
        frames_idx = np.linspace(0, vid_features.shape[0]-1, self.frames_limit).astype('int')
        vid_features = vid_features[frames_idx]
      testing_features[key] = vid_features
    
    bleu = []
    meteor = []
    for beam_width in beam_width_values:
      print("!!",beam_width)
      bleu_temp = []
      meteor_temp = []
      for key, features in testing_features.items():
        captions = self.__generate_captions_optimized(features, beam_width)[0]
        bleu_score = self.get_bleu_score(testing_sents[key], captions)
        meteor_score = self.get_meteor_score(testing_sents[key], captions)
        bleu_temp.append(bleu_score)
        meteor_temp.append(meteor_score)
      bleu.append(np.average(bleu_temp))
      meteor.append(np.average(meteor_temp))
    return bleu, meteor