from string import punctuation, ascii_lowercase

low_letters = {
    'english': ascii_lowercase,
    'russian': 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
}


def only_letters_pattern(lang='english'):
    try:
        return '^[' + low_letters[lang] + ']+$'
    except KeyError:
        print("Couldn't find {} language in only_letters_pattern".format(lang))