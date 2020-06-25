
import labeler
import experiment
import collections
import statistics
import pandas as pd 

model_path = './gpu_attention.model'

model = labeler.SequenceLabeler.load(model_path)

config = model.config
predictions_cache = {}

id2label = collections.OrderedDict()
for label in model.label2id:
	id2label[model.label2id[label]] = label


def get_complex_words(tokenised_string):

	dataframe = pd.DataFrame()
	dataframe['word'] = tokenised_string
	dataframe['binary'] = 'N'
	dataframe.to_csv('./'+'complex_word'+'.txt', sep = '\t',index=False, header=False, quotechar=' ')

	sentences_test = experiment.read_input_files('./complex_word.txt')
	batches_of_sentence_ids = experiment.create_batches_of_sentence_ids(sentences_test, config["batch_equal_size"], config['max_batch_size'])

	for sentence_ids_in_batch in batches_of_sentence_ids:
		batch = [sentences_test[i] for i in sentence_ids_in_batch]
		cost, predicted_labels, predicted_probs = model.process_batch(batch, is_training=False, learningrate=0.0)
	try:
		assert(len(sentence_ids_in_batch) == len(predicted_labels))
	except:
		print('cw error')

	prob_labels = predicted_probs[0]
	probability_list = []
	for prob_pair in prob_labels:
		probability_list.append(prob_pair[1])

	return probability_list


def get_complexities(indexes, tokenized_sentence):
	
	probabilities = get_complex_words(tokenized_sentence)

	word_probs = [probabilities[each_index] for each_index in indexes]
	
	return float(sum(word_probs))/len(word_probs)
	


def get_synonym_complexities(synonyms, tokenized, index):
	
	word_complexities = []
	
	for entry in synonyms:
		#index list for multi word replacements 
		indexes = []
		#create copy of original token list
		tokenized_sentence = tokenized.copy()

		
		del tokenized_sentence[index]
		#if synonym contains multiple words we calculate average complexity of words
		for i,word in enumerate(entry):
			#insert words 
			tokenized_sentence.insert((index + i), word)
			#append new indexes
			indexes.append(index+i)

		prob = get_complexities(indexes, tokenized_sentence)
		
		word_complexities.append(prob)

	return word_complexities

