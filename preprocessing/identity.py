from abstract_classes import TextPreprocessing
from preprocessing import Tokenizer, PrepLowerCase, FilterByRegex
from nltk import word_tokenize
from constants import only_letters_pattern


class PrepIdentity(TextPreprocessing):

    def __init__(self, lang='english'):
        self.prep_items = [
            PrepLowerCase(),
            Tokenizer(word_tokenize),
            FilterByRegex(only_letters_pattern(lang=lang))
        ]
        self.construct_mapping()
