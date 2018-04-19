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
		tags.append([(token['word'], token['pos'], token['characterOffsetBegin'], token['characterOffsetEnd'], token['index']) for token in sent['tokens']])
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

def check_spelling(dicc, parsed_sentences, orig_text,verbose=True):
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

def subject_verb_agreement_check(parsed_sentences, parsed_constituents, sentences_indexes, original_text):
	subject_verb_agreement_errors = list()
	vbz_agreement = ['EX', 'NN','NNP','PRP','WDT']
	vbp_agreement = ['EX','NNS','NNPS','PRP','WDT']
	vb_agreement_exceptions = ['DT','WP']


	for s_index, sent in enumerate(parsed_sentences):
		dependencies = parsed_constituents['sentences'][s_index]['basicDependencies']
		#print(dependencies)
		#print("Current Sentence")
		#print(sent)
		for t_index, token in enumerate(sent):
			#print(token)
			if(token[1]=='VBZ' or token[1]=='VBP'):
				#print("VBZ OR VBP")
				#print(token[t_index-1],sent[s_index])
				#print("Sentence: ", normal_sentences[s_index])
				current_pos = token[1]
				verb_index = token[4]
				verb_governor = [x for x in dependencies if x['dependent']== verb_index][0]['governor']
				verb_subject = [y for y in [x for x in dependencies if x['governor']==verb_governor] if y['dep']=='nsubj']
				subject_dependent = verb_subject[0]['dependent'] if len(verb_subject) > 0 else 0

				subject_token = [x for x in sent if x[4]==subject_dependent] 
				subject_pos = subject_token[0][1] if len(subject_token) > 0 else 'NaN'
				s_start = sentences_indexes['indices'][s_index][0]
				s_end = sentences_indexes['indices'][s_index][1]
				error_sentence = original_text[s_start : s_end]
				#print(error_sentence)
				if(subject_pos!='NaN'):
					if(current_pos=='VBZ' and subject_pos not in vbz_agreement and subject_pos not in string.punctuation):
						if(subject_pos not in vb_agreement_exceptions):
							subject_verb_agreement_errors.append((s_index,error_sentence,subject_token[0][0],subject_pos,token[0],token[1]))
					if(current_pos=='VBP' and subject_pos not in vbp_agreement and subject_pos not in string.punctuation):
						if(subject_pos not in vb_agreement_exceptions):
							subject_verb_agreement_errors.append((s_index,error_sentence,subject_token[0][0],subject_pos,token[0],token[1]))

	#print("Token: ", token, verb_index)
	#print("Verb governor: ", verb_governor)
	#print("verb subject: ", verb_subject)
	#print("subject dependent: ", subject_dependent)
	#print("Subject token: ", subject_token)
	#print("Subject pos: ", subject_pos)
	#print()


	return subject_verb_agreement_errors
def calculate_grade(scale, actual_score, grading_system):
	"""
		scale = an array with the scale distribution [1,2,3,4,5]
		actual_score = is the score assigned after distribution
		grading_system= 1to5 or 4to0
	"""
	g_result = 0
	if(grading_system=="1to5"):
		g_result = 1 if actual_score<=scale[0] else 2 if actual_score<=scale[1] else 3 if actual_score<=scale[2] else 4 if actual_score<=scale[3] else 5
	elif(grading_system=="4to0"):
		g_result = 0 if actual_score<=scale[0] else 1 if actual_score<=scale[1] else 2 if actual_score<=scale[2] else 3 if actual_score<=scale[3] else 4
	elif(grading_system=="5to1"):
		g_result = 5 if actual_score<=scale[0] else 4 if actual_score<=scale[1] else 3 if actual_score<=scale[2] else 2 if actual_score<=scale[3] else 1
	return g_result


def process_essays(essay_key):
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
	agreement_error_list = list()
	agreement_error_num = list()
	agreement_error_num_rate = list()
	print("***************Processing Files*********************")
	for essay in essay_key['essay']:

		parsed_constituents = constituency_parse(parser, essay, return_parse_obj=True)
		output =length_annotation(parsed_constituents, essay, verbose=False)
		words_count.append(output['words'])
		sentences_count.append(output['num_sentences'])
		constituents_count.append(output['num_constituents'])	
		result = check_spelling(dicc, pos_tags(pos_tagger, essay), essay, verbose=False)
		errors.append(result['num'])
		#errors_words.append(result['indices'])
		mispelling_rate.append((result['num']*100)/output['words'])
		# POS tagging before processing subject and verb agreement
		pos_tagged_sentences = pos_tags(pos_tagger, essay)
		# Sentences indexes to verify if encountered errors are not false positives
		sentence_indexes = num_sentence_annotation(parsed_constituents, essay, verbose=False)
		# Check verb and subject agreement
		s_v_agreement = subject_verb_agreement_check(pos_tagged_sentences, parsed_constituents, sentence_indexes, essay)
		agreement_error_list.append(s_v_agreement)
		agreement_error_num.append(len(s_v_agreement))
		agreement_error_num_rate.append(len(s_v_agreement)*100/output['num_sentences'])

	essay_key['words'] = words_count
	essay_key['sentences'] = sentences_count
	essay_key['constituents'] = constituents_count
	essay_key['mispelling words'] = errors
	#essay_key['wrong words'] = errors_words
	essay_key['mispelling rate'] = mispelling_rate
	essay_key['subject_verb_agreement'] = agreement_error_list
	essay_key['agreement_errors'] = agreement_error_num
	essay_key['agreement_errors_rate'] = agreement_error_num_rate

	"""essay_key = pd.read_csv('../output/result.csv', sep=',')

	print(essay_key.head())

	"""

	
	# Calculate the standard deviation for each column
	std_words = np.std(essay_key['words'])
	std_sentences = np.std(essay_key['sentences'])
	std_constituents = np.std(essay_key['constituents'])
	std_mispelling_words = np.std(essay_key['mispelling words'])
	std_mispelling_words_rate = np.std(essay_key['mispelling rate'])
	std_agreement_errors = np.std(essay_key['agreement_errors'])
	std_agreement_errors_rate = np.std(essay_key['agreement_errors_rate'])

	# Get the mean for each column
	mean_words = np.mean(essay_key['words'])
	mean_sentences = np.mean(essay_key['sentences'])
	mean_constituents = np.mean(essay_key['constituents'])
	mean_mispelling_words = np.mean(essay_key['mispelling words'])
	mean_mispelling_words_rate = np.mean(essay_key['mispelling rate'])
	mean_agreement_errors = np.mean(essay_key['agreement_errors'])
	mean_agreement_errors_rate = np.mean(essay_key['agreement_errors_rate'])
	#print("STDS")
	#print(std_words, std_sentences, std_constituents,std_mispelling_words)

	#print("MEANS")
	#print(mean_words, mean_sentences, mean_constituents, mean_mispelling_words, mean_mispelling_words_rate)

	#calculate score distribution formula

	#variables
	z_words = list()
	z_sentences = list()
	z_constituents = list()
	z_mispelling_words = list()
	z_mispelling_words_rate = list()
	z_agreement_errors = list()
	z_agreement_errors_rate = list()
	#apply the distribution for every column
	for index, row in essay_key.iterrows():
		z_words.append((row['words']-mean_words)/std_words)
		z_sentences.append((row['sentences']-mean_sentences)/std_sentences)
		z_constituents.append((row['constituents']-mean_constituents)/std_constituents)
		z_mispelling_words.append((row['mispelling words']-mean_mispelling_words)/std_mispelling_words)
		z_mispelling_words_rate.append((row['mispelling rate']-mean_mispelling_words_rate)/std_mispelling_words_rate)
		z_agreement_errors.append((row['agreement_errors']-mean_agreement_errors)/std_agreement_errors)
		z_agreement_errors_rate.append((row['agreement_errors_rate']-mean_agreement_errors_rate)/mean_agreement_errors_rate)

	essay_key['z_words'] = z_words
	essay_key['z_sentences'] = z_sentences
	essay_key['z_constituents'] = z_constituents
	essay_key['z_mispelling_words'] =z_mispelling_words
	essay_key['z_mispelling_words_rate'] = z_mispelling_words_rate
	essay_key['z_agreement_errors'] = z_agreement_errors
	essay_key['z_agreement_errors_rate'] = z_agreement_errors_rate

	#calculate scale to assign scoring
	#scale, the number of possible scores to be assigned
	scale = 5
	max_z_words = np.amax(z_words)
	min_z_words = np.amin(z_words)

	pivot = (max_z_words-min_z_words)/scale
	#scores for words from 1 to 5
	scale_words = [min_z_words+pivot,min_z_words+(pivot*2),min_z_words+(pivot*3),min_z_words+(pivot*4),min_z_words+(pivot*5)]

	#score for sentences from 1 to 5
	max_z_sentences = np.amax(z_sentences)
	min_z_sentences = np.amin(z_sentences)
	pivot = (max_z_sentences-min_z_sentences)/scale

	scale_sentences = [min_z_sentences+pivot, min_z_sentences+(pivot*2),min_z_sentences+(pivot*3),min_z_sentences+(pivot*4),min_z_sentences+(pivot*5)]

	#score for constituents from 1 to 5
	max_z_constituents = np.amax(z_constituents)
	min_z_constituents = np.amin(z_constituents)
	pivot = (max_z_constituents-min_z_constituents)/scale

	scale_constituents = [min_z_constituents+pivot, min_z_constituents+(pivot*2),min_z_constituents+(pivot*3),min_z_constituents+(pivot*4),min_z_constituents+(pivot*5)]

	#score for mispelling words from 4 down to 0
	max_z_mispelling = np.amax(z_mispelling_words)
	min_z_mispelling = np.amin(z_mispelling_words)
	pivot = (max_z_mispelling-min_z_mispelling)/scale

	scale_mispelling = [min_z_mispelling+pivot, min_z_mispelling+(pivot*2),min_z_mispelling+(pivot*3),min_z_mispelling+(pivot*4),min_z_mispelling+(pivot*5)]

	#score for mispelling words rate from 4 down to 0
	max_z_mispelling_rate = np.amax(z_mispelling_words_rate)
	min_z_mispelling_rate = np.amin(z_mispelling_words_rate)
	pivot = (max_z_mispelling_rate-min_z_mispelling_rate)/scale

	scale_mispelling_rate = [min_z_mispelling_rate+pivot, min_z_mispelling_rate+(pivot*2),min_z_mispelling_rate+(pivot*3),min_z_mispelling_rate+(pivot*4),min_z_mispelling_rate+(pivot*5)]

	#score for agreement errors and rates from 5 down to 1
	max_z_agreement_errors = np.amax(z_agreement_errors)
	min_z_agreement_errors = np.amin(z_agreement_errors)
	pivot = (max_z_agreement_errors - min_z_agreement_errors) / scale

	scale_agreement_errors = [min_z_agreement_errors+pivot, min_z_agreement_errors+(pivot*2),min_z_agreement_errors+(pivot*3),min_z_agreement_errors+(pivot*4),min_z_agreement_errors+(pivot*5)]

	max_z_agreement_errors_rate = np.amax(z_agreement_errors_rate)
	min_z_agreement_errors_rate = np.amin(z_agreement_errors_rate)
	pivot = (max_z_agreement_errors_rate-min_z_agreement_errors_rate) / scale 

	scale_agreement_errors_rate = [min_z_agreement_errors_rate+pivot, min_z_agreement_errors_rate+(pivot*2),min_z_agreement_errors_rate+(pivot*3),min_z_agreement_errors_rate+(pivot*4),min_z_agreement_errors_rate+(pivot*5)]

	print(scale_words)
	print(scale_sentences)
	print(scale_constituents)
	print(scale_mispelling)
	print(scale_mispelling_rate)

	#assign scores
	score_words = list()
	score_sentences = list()
	score_constituents = list()
	score_mispelling = list()
	score_mispelling_rate = list()
	score_agreement_errors = list()
	score_agreement_errors_rate = list()

	#apply the distribution for every column
	for index, row in essay_key.iterrows():
		score_words.append(calculate_grade(scale_words, row['z_words'], "1to5"))
		score_sentences.append(calculate_grade(scale_sentences, row['z_sentences'], "1to5"))
		score_constituents.append(calculate_grade(scale_constituents, row['z_constituents'], "1to5"))
		score_mispelling.append(calculate_grade(scale_mispelling, row['z_mispelling_words'], "4to0"))
		score_mispelling_rate.append(calculate_grade(scale_mispelling_rate, row['z_mispelling_words_rate'], "4to0"))
		score_agreement_errors.append(calculate_grade(scale_agreement_errors, row['z_agreement_errors'], "5to1"))
		score_agreement_errors_rate.append(calculate_grade(scale_agreement_errors_rate, row['z_agreement_errors_rate'],"5to1"))

	essay_key['words grade'] = score_words
	essay_key['sentences grade'] = score_sentences
	essay_key['constituents grade'] = score_constituents
	essay_key['spelling grade'] = score_mispelling
	essay_key['spelling grade rate'] = score_mispelling_rate
	essay_key['agreement_errors_grade'] = score_agreement_errors
	essay_key['agreement_errors_rate_grade'] = score_agreement_errors_rate

	essay_key['a'] = (essay_key['words grade'] + essay_key['sentences grade'] +essay_key['constituents grade'])/3
	essay_key['b'] = (essay_key['spelling grade'] + essay_key['spelling grade rate']) /2
	essay_key['c'] = (essay_key['z_agreement_errors']+essay_key['z_agreement_errors_rate'])/2
	essay_key['2ab'] = (2*essay_key['a'])-essay_key['b']+essay_key['c']
	return essay_key
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
	essay_key_df = pd.read_csv('../input/training/index2.csv', sep=',')

	essay_key_df = process_essays(essay_key_df)
	

	high_grades = essay_key_df.loc[essay_key_df['grade'] == 'high']
	low_grades = essay_key_df.loc[essay_key_df['grade'] == 'low']

	high_cut = (np.amin(high_grades['2ab'])+np.amax(low_grades['2ab']))/2

	print("HIGH_CUT")
	print(high_cut)

	print('Grading Testing Essays')

	testing_df = pd.read_csv('../input/testing/index22.csv', sep=',')
	testing_df = process_essays(testing_df)

	testing_grades = list()

	for g_2ab in testing_df['2ab']:
		if(g_2ab>= high_cut):
			testing_grades.append('high')
		else:
			testing_grades.append('low')
	testing_df['final_grade'] = testing_grades

	# Output csv
	testing_df[['filename','grade','words', 'sentences', 'constituents','mispelling words', 'mispelling rate','agreement_errors','agreement_errors_rate','z_words', 'z_sentences','z_constituents','z_mispelling_words','z_mispelling_words_rate','z_agreement_errors','z_agreement_errors_rate','words grade', 'sentences grade','constituents grade','spelling grade','spelling grade rate','agreement_errors_grade','agreement_errors_rate_grade','a','b','c','2ab','final_grade']].to_csv(
		'../output/final_grades.csv', 
		index=False)
	print("***************Process Finished!*********************")
	

