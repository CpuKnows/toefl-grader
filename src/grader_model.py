import json
import re
import pandas as pd
import nltk
import numpy as np

from nltk.tree import Tree
# This uses corenlp server! Will need to alter code if using JAR files directly
# https://stanfordnlp.github.io/CoreNLP/corenlp-server.html
from nltk.parse.corenlp import CoreNLPParser
from nltk.tag.stanford import CoreNLPTagger, CoreNLPPOSTagger, CoreNLPNERTagger

# For spelling checker
from enchant import Dict as dictionary
import seaborn as sns
import matplotlib.pyplot as plt
import string

# Functions

# Altered behavior of NLTK so CoreNLP performs sentence splits
def constituency_parse(parser, sentences, return_parse_obj=False):
	"""Creates parse strings for each sentence.  
	Each parse string can be fed into Tree.fromstring() to create NLTK Tree objects.

	parser (CoreNLPParser): parser to parse sentences
	sentences (str): essay text
	return_parse_obj (bool): return parse object or string of trees
	RETURNS (list): a list of parses in string form
	"""
	default_properties = {'outputFormat': 'json', 
						  'annotators': 'tokenize,pos,lemma,ssplit,parse'}
	parsed_data = parser.api_call(sentences, properties=default_properties)
	if return_parse_obj:
		return parsed_data
	else:
		parses = list()
		for parsed_sent in parsed_data['sentences']:
			parse = parsed_sent['parse']
			# Compress whitespace
			parse = re.sub('[\s]+', ' ', parse)
			parses.append(parse)
		return parses

def pos_tags(tagger, sentences):
	"""Tags sentences with POS tags. Returns a list of (word, tag, start index, end index) tuples

	tagger (CoreNLPTagger): a tagger to tag sentences
	RETURNS (list): list of (word, tag) tuples
	"""
	default_properties = {'annotators': 'tokenize,ssplit,pos'}
	tagged_data = tagger.api_call(sentences, properties=default_properties)
	
	tags = list()
	for sent in tagged_data['sentences']:
		tags.append([(token['word'], token['pos'], token['characterOffsetBegin'], token['characterOffsetEnd']) for token in sent['tokens']])
	return tags

def tree_to_str(trees):
	"""Joins a list of trees in string form"""
	return ' '.join(trees)

def str_to_trees(tree_str):
	"""Splits a string into a list of trees in string form"""
	d = "(ROOT"
	return  [(d+sent).strip() for sent in tree_str.split(d) if sent]

def num_sentence_annotation(parsed_data, orig_text, verbose=True):
	sentences = dict()
	sentences['indices'] = list()
	
	if verbose:
		print('Format: (Start index, end index) Sentence')
		print()
		
	for sent in parsed_data['sentences']:
		start_offset = sent['tokens'][0]['characterOffsetBegin']
		end_offset = sent['tokens'][-1]['characterOffsetEnd']
		sentences['indices'].append((start_offset, end_offset))
		if verbose:
			print((start_offset, end_offset), orig_text[start_offset:end_offset])
			print()
		
	sentences['num'] = len(parsed_data['sentences'])
	if verbose:
		print('Num sentence:', len(parsed_data['sentences']))
		print()
	return sentences

def length_annotation(parsed_data, orig_text, verbose=True):
	sentences = dict()
	sentences['indices'] = list()
	sentences['constituents'] = list()
	words = nltk.word_tokenize(orig_text)
	nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
	sentences['words'] = len([w for w in words if nonPunct.match(w)])
	
	if verbose:
		print('Format: (Start index, end index) Sentence')
		print()
		
	for sent in parsed_data['sentences']:
		start_offset = sent['tokens'][0]['characterOffsetBegin']
		end_offset = sent['tokens'][-1]['characterOffsetEnd']
		sentences['indices'].append((start_offset, end_offset))
		
		# Get the height of the parsed tree 
		t = Tree.fromstring(sent['parse'])
		sentences['constituents'].append(t.height())
		if verbose:
			print((start_offset, end_offset), orig_text[start_offset:end_offset])
			print()
		
	sentences['num_sentences'] = len(parsed_data['sentences'])
	sentences['num_constituents'] = sum(sentences['constituents'])
	if verbose:
		print('Num sentence:', len(parsed_data['sentences']))
		print()
	return sentences

def sentence_sanity_check(sentence_dict):
	error = False
	
	if sentence_dict['num'] != len(sentence_dict['indices']):
		print('Number of sentences does not match')
		error = True
		
	prev = 0
	for i,j in enumerate(sentence_dict['indices']):
		if j[0] >= j[1]:
			print('Sentence', i, ': Start/end indices overlap')
			error = True
		if prev >= j[0] and prev != 0:
			print('Sentence', i, ': Previous end index overlaps start index')
			error = True
		if j[0] - prev > 1:
			print('Sentence', i, ': Is gap between sentences > 1 character/space?')
			error = True
		prev = j[1]
	
	if not error:
		print('No errors')

def find_word(orig_text, word):
	matches = list()
	for m in re.finditer(word, orig_text, flags=re.I):
		#print(m)
		if m.span(0)[0] < 10:
			before_context = (' ' * (10-m.span(0)[0])) + orig_text[0:m.span(0)[0]]
		else:
			before_context = orig_text[m.span(0)[0]-10 : m.span(0)[0]]
		
		if len(orig_text) - m.span(0)[1] < 10:
			#print(orig_text[m.span(0)[1]:])
			#print(' ' * (len(orig_text) - m.span(0)[1]))
			after_context = orig_text[m.span(0)[1]:] + (' ' * (len(orig_text) - m.span(0)[1]))
		else:
			after_context = orig_text[m.span(0)[1] : m.span(0)[1]+10]
			
		matches.append((m.span(0), '...' + before_context + orig_text[m.span(0)[0]:m.span(0)[1]] + after_context + '...'))
		
	if len(matches) == 1:
		return matches[0][0]
	else:
		for i,m in enumerate(matches):
			print('Index:', i, '-', m)
			
		choice = int(input('Choose a match index (number) or -1 for all: '))
		if choice == -1:
			return [m[0] for m in matches]
		else:
			return matches[choice][0]

def check_spelling(dictionary, parsed_sentences, orig_text,verbose=True):
	false_positives = ['-LRB-','-RRB-',"''",',','\'s','\\','n\'t',':'';','!','?','\'m','\'d','\'\'','??','-','(',')','\'ve','\'','\'re','!','e.g.','[',']','_','>>','>','<','<<','!!','"','``']
	"""Check the spelling of tagged words on an essay
	and return the list of misspelling words with their tags 
	and indexes of begining and end of those words"""
	wrong_words = dict()
	wrong_words['indices'] = list()
	
	if verbose:
		print('Format: (Start index, end index) word')
		print()
	
	found_words = list()
	for sentence in parsed_sentences:
		for w_tuple in sentence:
			word = w_tuple[0]
			if (word not in string.punctuation and word not in false_positives):
				if dicc.check(word) is False:
					tag = w_tuple[1]
					start_offset = w_tuple[2]
					end_offset = w_tuple[3]
					if word not in found_words:
						wrong_words['indices'].append((word, tag, start_offset, end_offset))
						found_words.append(word)
					if verbose:
						print((start_offset, end_offset), orig_text[start_offset:end_offset])
						print()
	wrong_words['num'] = len(wrong_words['indices'])
	if verbose:
		print('Num wrong words:', len(wrong_words['indices']))
		print()
	return wrong_words


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
	# Careful! CoreNLPTagger, CoreNLPPOSTagger, and CoreNLPNERTagger will all be replaced in the next NLTK version (3.2.6)
	parser = CoreNLPParser(url='http://localhost:9000')
	#pos_tagger = CoreNLPPOSTagger(url='http://localhost:9000')
	#ner_tagger = CoreNLPNERTagger(url='http://localhost:9000')
	pos_tagger = CoreNLPTagger(tagtype='pos', url='http://localhost:9000')
	ner_tagger = CoreNLPTagger(tagtype='ner', url='http://localhost:9000')
	# Parser
	parser = CoreNLPParser(url='http://localhost:9000')
	# Get essays
	essay_key = pd.read_csv('../input/training/index.csv', sep=';')

	essays = list()
	for filename in essay_key['filename']:
		with open('../input/training/essays/'+filename, 'r') as f:
			essays.append(f.read().strip())
			
	essay_key['essay'] = essays


	# Parse for each essay
	dicc = dictionary("en_US")
	errors = list()
	#errors_words = list()
	words_count = list()
	sentences_count= list()
	constituents_count= list()
	mispelling_rate = list()

	print("***************Processing Files*********************")
	for essay in essay_key['essay']:
		output =length_annotation(constituency_parse(parser, essay, return_parse_obj=True), essay, verbose=False)
		words_count.append(output['words'])
		sentences_count.append(output['num_sentences'])
		constituents_count.append(output['num_constituents'])	
		result = check_spelling(dicc, pos_tags(pos_tagger, essay), essay, verbose=False)
		errors.append(result['num'])
		#errors_words.append(result['indices'])
		mispelling_rate.append((result['num']*100)/output['words'])

	essay_key['words'] = words_count
	essay_key['sentences'] = sentences_count
	essay_key['constituents'] = constituents_count
	essay_key['mispelling words'] = errors
	#essay_key['wrong words'] = errors_words
	essay_key['mispelling rate'] = mispelling_rate

	# Output csv
	essay_key[['filename','grade','words', 'sentences', 'constituents','mispelling words', 'mispelling rate']].to_csv(
		'../output/result.csv', 
		index=False)

	print("***************Process Finished!*********************")

