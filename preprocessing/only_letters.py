from nltk.tokenize import word_tokenize
from abstract_classes.text_prep import TextPreprocessing
from constants import only_letters_pattern
from preprocessing.preprocessing_items import *


# Removing stopwords and everything except letters, making text lower
class OnlyLetters(TextPreprocessing):

    def __init__(self, lang='english'):
        try:
            self.prep_items = [
                PrepLowerCase(),
                Tokenizer(word_tokenize),
                PrepRemoveStopWords(lang=lang),
                FilterByRegex(only_letters_pattern(lang=lang))
            ]
        except KeyError:
            print("Couldn't find letters for {} language in constants.py".format(lang))
        self.construct_mapping()
