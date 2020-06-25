import nltk
import complex_word
import spacy
import json
import urllib
import plural
import verb 
import pandas as pd
import helper_functions as hf
import string
import inflect

class Sentence:

	def __init__(self, tokenized, threshold, ignore_list):
		self.threshold = threshold
		self.tokenized = tokenized
		self.indexes = list(enumerate(self.tokenized))
		self.pos_tags = nltk.pos_tag(self.tokenized)
		if ignore_list == []:
			self.ignore_index = [c for (a,b),(c,d) in zip(self.pos_tags, self.indexes) if 'P' in b]
		else:
			self.ignore_index = ignore_list

		#print(complex_word.get_complex_words(self.tokenized))


		self.complex_words = [(a,b) for a,b in list(zip([a for a,b in self.indexes], complex_word.get_complex_words(self.tokenized))) if b > self.threshold]
		self.complex_words = [(a,b) for a,b in self.complex_words if a not in self.ignore_index]
		self.complex_words = sorted(self.complex_words, key = lambda x: x[1], reverse=True)
	

	def add_ignore(self, item):
		self.ignore_index.append(item)
	

	def make_simplification(self, synonym, index):

		tokens = self.tokenized
	
		del tokens[index]

		for i,word in enumerate(synonym):
			tokens.insert((index + i), word)
			self.add_ignore(index)
			

		self.tokenized = tokens
	
		self.indexes = list(enumerate(self.tokenized))
		self.pos_tags = nltk.pos_tag(self.tokenized)

		self.complex_words = [(a,b) for a,b in list(zip([a for a,b in self.indexes], complex_word.get_complex_words(self.tokenized))) if b > self.threshold]
		self.complex_words = [(a,b) for a,b in self.complex_words if a not in self.ignore_index]
		self.complex_words = sorted(self.complex_words, key = lambda x: x[1], reverse=True)
		

class Word:

	def __init__(self, sentence_object, index):

		pos_tags = sentence_object.pos_tags
		self.token_sent = sentence_object.tokenized
		self.pos_sent = nltk.pos_tag(self.token_sent)
		self.word = pos_tags[index][0]
		self.pos = pos_tags[index][1]
		self.index = index
		self.synonyms = None
		self.tense = None
		self.lemma = None
		self.is_plural = True if (self.pos == 'NNS') else False


	def get_synonyms(self):
		

		spacy_module = spacy.load('en_core_web_sm')

		doc = spacy_module("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in self.token_sent]).strip())

		for token in doc:
			if str(token) == self.word:
				self.lemma = token.lemma_

		if self.lemma==None:
			self.synonyms = []
			return
		

		import requests

		##6c6bbfe357c61dcc40b628419778ebd7

		##ce218b46b8d46a30bebc843f4da120d8
		
		r = requests.get(url='http://words.bighugelabs.com/api/2/6c6bbfe357c61dcc40b628419778ebd7/'+self.lemma+'/json')
		if r.status_code != 404:

			try:
				#print("----------")
				#print(r.json())

				if type(r.json())==list:
					self.synonyms = r.json()
				elif 'V' in self.pos:
					self.synonyms = r.json()["verb"]["syn"]
					self.tense = verb.verb_tense(self.word)
				elif 'N' in self.pos:
					try:
						self.synonyms = r.json()["noun"]["syn"] + r.json()["noun"]["sim"]
					except:
						self.synonyms = r.json()["noun"]["syn"]
				elif 'J' in self.pos:
					try:
						self.synonyms = r.json()["adjective"]["syn"] + r.json()["adjective"]["sim"]
					except:
						self.synonyms = r.json()["adjective"]["syn"]
				elif 'RB' in self.pos:
					try:
						self.synonyms = r.json()["adverb"]["syn"] + r.json()["adverb"]["sim"]
					except:
						self.synonyms = r.json()["adverb"]["syn"]
			except:
				total_list = []
				for pos in r.json():
					for type_ in r.json()[pos]:
						total_list.append(r.json()[pos][type_])
				self.synonyms = [item for sublist in total_list for item in sublist]

		if self.synonyms==None:
			return
		self.synonyms = [x.split(' ') for x in self.synonyms]

		temp_set = []

		for word in self.synonyms:
			temp_set.append(word[0])

		temp_set = set(temp_set)

		temp_set = [[x] for x in temp_set]

		self.synonyms = temp_set


		if self.is_plural == True:
			p = inflect.engine()
			all_synonyms = []
			for synonym in self.synonyms:
				new_synonyms = []
				for word in synonym:
					if p.singular_noun(word) is False:
						new_synonyms.append(plural.noun_plural(word))
					else:
						new_synonyms.append(word)
				all_synonyms.append(new_synonyms)
					

			self.synonyms = all_synonyms

		if self.tense != None:
			tense_synonyms = []
			for x in self.synonyms:
				multi_word = []
				for element in x:
					try:
						multi_word.append((verb.verb_conjugate(element, tense=self.tense, negate=False)))
					except:
						multi_word.append(element)
				tense_synonyms.append(multi_word)
				
			self.synonyms = tense_synonyms

		
			

	def get_ranked_synonyms(self):
		if len(self.synonyms) > 1:
			synonym_scores = pd.DataFrame()
			synonym_scores['synonyms'] = self.synonyms
			synonym_scores['sem_sim'] = hf.get_elmo_score(self.synonyms, self.token_sent, self.index)
			synonym_scores['complexity'] = complex_word.get_synonym_complexities(self.synonyms, self.token_sent, self.index)
			synonym_scores['grammaticality'] = hf.get_gram_score(self.synonyms, self.token_sent, self.pos_sent, self.index)
			#filtering process
			synonym_scores = synonym_scores[synonym_scores['sem_sim']<0.15]
			synonym_scores = synonym_scores.sort_values(by=['complexity'])

			
			try:
				top_synomym = synonym_scores.synonyms.values[0]
				
			except:
				return [self.word]

			return top_synomym
		else:
			return [self.word]


	def get_synonym_complexities(self):
		if len(self.synonyms) > 1:
			synonym_scores = pd.DataFrame()
			synonym_scores['synonyms'] = self.synonyms
			synonym_scores['sem_sim'] = hf.get_elmo_score(self.synonyms, self.token_sent, self.index)
			synonym_scores['complexity'] = complex_word.get_synonym_complexities(self.synonyms, self.token_sent, self.index)
			synonym_scores['grammaticality'] = hf.get_gram_score(self.synonyms, self.token_sent, self.pos_sent, self.index)
			
			#filtering process, return top word?
			synonym_scores = synonym_scores[synonym_scores['sem_sim']<0.15]
			synonym_scores = synonym_scores.sort_values(by=['complexity'])

			return list(zip(synonym_scores['synonyms'].values,synonym_scores['complexity'].values))
		else:
			return None
		

	def get_synonym_dataframe(self):
		if len(self.synonyms) > 1:
			synonym_scores = pd.DataFrame()
			synonym_scores['synonyms'] = self.synonyms
			synonym_scores['sem_sim'] = hf.get_elmo_score(self.synonyms, self.token_sent, self.index)
			synonym_scores['complexity'] = complex_word.get_synonym_complexities(self.synonyms, self.token_sent, self.index)
			synonym_scores['grammaticality'] = hf.get_gram_score(self.synonyms, self.token_sent, self.pos_sent, self.index)
			
		
			synonym_scores = synonym_scores[synonym_scores['sem_sim']<0.3]
			#Can filter to only replace with words of lower threshold complexity
			#synonym_scores = synonym_scores[synonym_scores['complexity']<0.6]
			synonym_scores = synonym_scores[synonym_scores['grammaticality']==1]
			synonym_scores['combo'] = synonym_scores['sem_sim'] + synonym_scores['complexity']
			synonym_scores = synonym_scores.sort_values(by=['combo'])


			return synonym_scores
		else:
			return None

		
		