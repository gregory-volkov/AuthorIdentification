from nltk.corpus import stopwords
import re
from abstract_classes.text_prep import TextPreprocessingItem
from constants import punctuation


# Makes string lowercase
class PrepLowerCase(TextPreprocessingItem):

    def mapping(self, text):
        return text.lower()


# Abstract class representing removing substrings using some pattern
class PrepRemoveByRegex(TextPreprocessingItem):

    _regex = None

    def mapping(self, text):
        text = self._regex.sub('', text)
        return text


# Removing punctuation marks
class PrepRemovePunctuation(PrepRemoveByRegex):

    def __init__(self, marks=punctuation):
        pattern = r"[{}]".format(marks)
        self._regex = re.compile(pattern)


# Removing by white list
class PrepRemoveByBlackList(TextPreprocessingItem):

    black_list = None

    def mapping(self, tokens):
        return list(filter(lambda x: x not in self.black_list, tokens))


# Remove stopwords
class PrepRemoveStopWords(PrepRemoveByBlackList):

    def __init__(self, lang="english"):
        self.black_list = stopwords.words("english")


class Tokenizer(TextPreprocessingItem):

    def __init__(self, token_func):
        self.tokenizer = token_func

    def mapping(self, text):
        return self.tokenizer(text)


class FilterByRegex(TextPreprocessingItem):

    _regex = None

    def __init__(self, pattern):
        self._regex = re.compile(pattern)

    def mapping(self, tokens):
        return list(filter(self._regex.search, tokens))
