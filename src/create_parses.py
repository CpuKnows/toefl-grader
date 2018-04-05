###############################################################################
# File: create_parses.py
# Author: John Maxwell
# Date: 2018-04-04
#
# Create a file with constituency parses for each essay
###############################################################################
import re
import pandas as pd
import nltk

from nltk.tree import Tree
# This uses corenlp server! Will need to alter code if using JAR files directly
# https://stanfordnlp.github.io/CoreNLP/corenlp-server.html
from nltk.parse.corenlp import CoreNLPParser
from nltk.tag.stanford import CoreNLPTagger, CoreNLPPOSTagger, CoreNLPNERTagger

###############################################################################
# Functions
###############################################################################

# Altered behavior of NLTK so CoreNLP performs sentence splits
def constituency_parse(parser, sentences):
    """Creates parse strings for each sentence.  
    Each parse string can be fed into Tree.fromstring() to create NLTK Tree objects.

    parser (CoreNLPParser): parser to parse sentences
    RETURNS (list): a list of parses in string form
    """
    default_properties = {'outputFormat': 'json', 
    					  'annotators': 'tokenize,pos,lemma,ssplit,parse'}
    parsed_data = parser.api_call(sentences, properties=default_properties)
    
    parses = list()
    for parsed_sent in parsed_data['sentences']:
        parse = parsed_sent['parse']
        # Compress whitespace
        parse = re.sub('[\s]+', ' ', parse)
        parses.append(parse)
    return parses
        
def pos_tags(tagger, sentences):
    """Tags sentences with POS tags. Returns a list of (word, tag) tuples

    tagger (CoreNLPTagger): a tagger to tag sentences
    RETURNS (list): list of (word, tag) tuples
    """
    default_properties = {'annotators': 'tokenize,ssplit,pos'}
    tagged_data = tagger.api_call(sentences, properties=default_properties)
    
    tags = list()
    for sent in tagged_data['sentences']:
        tags.append([(token['word'], token['pos']) for token in sent['tokens']])
    return tags

def tree_to_str(trees):
    """Joins a list of trees in string form"""
    return ' '.join(trees)

def str_to_trees(tree_str):
    """Splits a string into a list of trees in string form"""
    d = "(ROOT"
    return  [(d+sent).strip() for sent in tree_str.split(d) if sent]


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
	# Parser
	parser = CoreNLPParser(url='http://localhost:9000')
	
	# Get essays
	essay_key = pd.read_csv('../data/essays_dataset/index.csv', sep=';')

	essays = list()
	for filename in essay_key['filename']:
	    with open('../data/essays_dataset/essays/'+filename, 'r') as f:
	        essays.append(f.read().strip())
	        
	essay_key['essay'] = essays

	# Constituency parse for each essay
	parsed_essays = list()
	num_sentences = list()
	for essay in essay_key['essay']:
		# Parse the essay
		trees = constituency_parse(parser, essay)
		# Number of sentences in the essay
		num_sentences.append(len(trees))
		# String of parsed sentences
		trees_str = tree_to_str(trees)
		parsed_essays.append(trees_str)

	essay_key['num_sentences'] = num_sentences
	essay_key['parsed_essay'] = parsed_essays

	# Output csv
	essay_key[['filename', 'num_sentences', 'parsed_essay']].to_csv(
		'../data/essays_dataset/index_with_parse.csv', 
		index=False)
