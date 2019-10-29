import sys
import argparse
import numpy as np 
from pliers.extractors.text_encoding import DirectSentenceExtractor,\
embedding_methods,DirectTextExtractorInterface

#
# processing parameers
#

def parseArguments():
    
    parser = argparse.ArgumentParser(description='Text encoding via Pliers')

    ## arguments about input/output
    
    parser.add_argument('--input',type=str,required=True, 
                        help = 'input file, each line is embedded separately (required) ')
    parser.add_argument('--output_prefix',type=str,required=True,
                        help = 'prefix for output files (including output path)')
    parser.add_argument('--output_type',type=str,default='one_vector_per_line',
                        help = 'output one_vector_per_line (default, one output file) or ' +
                        'one_vector_per_prefix (vectors for all prefixes of each line, one output file per line)')
    
    ## arguments pertaining to the embedding method

    parser.add_argument('--method', type=str, default='glove',
                         help = 'select from glove, word2vec, fasttext ' +
                            '(word embeddings, sentence embedding is average of (content) word embeddings), ' +
                            'or elmo, bert (sentence embeddings))')

    parser.add_argument('--dimensionality', type=int, default=300, 
                        help = 'embedding dimension (only for word embeddings, ignored otherwise)')

    parser.add_argument('--corpus', type=str, default='42B',
                        help = 'corpus used to create embedding (only used for glove, at present)')
    
    
#    parser.add_argument('--method_name', type=str, default='averageWordEmbedding',
#                         help = 'text encoding via word or sentence embedding. Note, ' + 
#                         'if the selected method is not average embedding, ' + 
#                         'e.g., Elmo or Bert, subsequentt arguments will be ' + 
#                         'neglected.')
    
    ## arguments that apply across methods and change their operation
    
    parser.add_argument('--content_only', type=bool, default=True,
                        help = 'whether only content words are used')
    parser.add_argument('--stopWords', type=list, default=None, 
                        help = 'remove stopwords, (use stopwords from NLTK, unless another file is provided)')
    parser.add_argument('--unk_vector', type=list, default=None,
                        help = 'particular vector to use for unknown words (default is all zeros) ')
    parser.add_argument('--binary', type=bool, default=False)

    
    args = parser.parse_args()
       
    return args

#######################################
#
# main function
#
#######################################

def main():

    #
    # parameter processing
    #
    
    arguments = parseArguments();

    inputFile    = arguments.input;
    outputPrefix = arguments.output_prefix;
    outputType   = arguments.output_type;
    
    method = arguments.method;
    embedding = method;
    if (method == "glove") | (method == "word2vec") | (method == "fasttext"):
        method    = 'averageWordEmbedding';
    elif method == "elmo":
        dimensionality = 1000;
    elif method == "bert":
        dimensionality = 1000;
    else:
         print("error: unknown method " + method); sys.exit(1);
        
    dimensionality = arguments.dimensionality;
    corpus         = arguments.corpus;

    content_only   = arguments.content_only;
    stopWords      = arguments.stopWords;
    unk_vector     = arguments.unk_vector;
    binary         = arguments.binary;
    if outputType == "one_vector_per_line":
        cbow = False;
    elif outputType == "one_vector_per_prefix":
        cbow = True;
    else:
        print("error: invalid outputType " + outputType); sys.exit(1);

    #
    # execution
    #

    lines = [line.rstrip('\n') for line in open(inputFile)]

    sys.exit(1);

    ## instantiate an extractor

    print("instantiating extractor (may take a while");
        
    extractor = DirectTextExtractorInterface(method=method,\
                                             embedding=embedding,\
                                             dimensionality=dimensionality,\
                                             corpus=corpus,\
                                             content_only=content_only,\
                                             binary = binary,\
                                             stopWords=stopWords,\
                                             unk_vector=unk_vector);

        
    ## grab all the lines in the input text file

    #lines = [for line in open 

    ##

    if outputType == "one_vector_per_line":
        # embed all lines at once
         vectors = textExtractor(extractor,method=method,\
                                     inputFile=inputFile,cbow=cbow)
        # save

    else:
        pass
        # loop over lines
