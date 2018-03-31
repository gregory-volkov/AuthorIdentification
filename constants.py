from string import punctuation, ascii_lowercase

low_letters = {
    'english': ascii_lowercase,
    'russian': 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
}

only_letters_pattern = r'^[\w]+$'
