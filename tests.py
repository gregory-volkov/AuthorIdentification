from abstract_classes.test import Test
from collections import OrderedDict


english_test_dicts = {

    "aus_king":
        {
            "data/books/austen/": [
                "abbey.txt",
                "persuasion.txt",
                "pride.txt"
            ],

            "data/books/king/": [
                "running.txt",
                "gunslinger.txt",
                "green.txt"
            ]
        },

    "orwell_tolk_aus":
        {
            "data/books/orwell/": [
                "burmese.txt",
                "catalonia.txt",
                "london_paris.txt",
                "animal_farm.txt"
            ],

            "data/books/tolkien/": [
                "ring_lord_1.txt",
                "ring_lord_2.txt",
                "hobbit.txt"
            ],

            "data/books/austen/": [
                "sense.txt",
                "pride.txt"
            ]
        },

    "orwell_king":
        {
            "data/books/orwell/": [
                "burmese.txt",
                "catalonia.txt",
                "london_paris.txt",
                "animal_farm.txt"
            ],

            "data/books/king/": [
                "running.txt",
                "gunslinger.txt",
                "green.txt"
            ]
        }
}

russian_test_dicts = {
    "tur_dos":
        {
            "data/books/turgenev/": [
                "fathers_sons.txt",
                "home_gentry.txt",
                "eve.txt"
            ],
            "data/books/dostoyevsky/": [
                "idiot_1.txt",
                "idiot_2.txt"
            ]
        },

    "push_ser_dos":
        {
            "data/books/pushkin/": [
                "captain_daughter.txt",
                "onegin_dubrov.txt"
            ],

            "data/books/serafimovich/": [
                "step.txt",
                "story_1.txt",
                "story_2.txt",
                "story_3.txt"
            ],

            "data/books/dostoyevsky/": [
                "idiot_1.txt",
                "idiot_2.txt"
            ]
        },
    "shol_tol":
        {
            "data/books/sholokhov/don_by_parts/": [
                str(i) + 'p.txt' for i in range(1, 9)
            ],
            "data/books/tolstoy/": [
                "war_peace_1_1.txt",
                "war_peace_1_2.txt",
                "war_peace_2_1.txt",
                "war_peace_2_2.txt"
            ]
        },


    "shol_ser":
        {
            "data/books/sholokhov/don_by_parts/": [
                str(i) + 'p.txt' for i in range(1, 9)
            ],
            "data/books/serafimovich/": [
                "story_3.txt",
                "story_2.txt",
                "story_1.txt",
                "step_iron.txt"
            ]
        }
}

english_test_dicts = {key: OrderedDict(value) for key, value in english_test_dicts.items()}
russian_test_dicts = {key: OrderedDict(value) for key, value in russian_test_dicts.items()}
eng_tests = [Test(path_dict, test_name) for test_name, path_dict in english_test_dicts.items()]
rus_tests = [Test(path_dict, test_name) for test_name, path_dict in russian_test_dicts.items()]
