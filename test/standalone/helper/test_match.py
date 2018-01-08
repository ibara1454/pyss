#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from pyss.util.match import *

mapper_1 = {
    'key1': [('pattern1_1', 'value1'), ('pattern1_2', 'value2')],
    'key2': [('pattern2_1', 'value1')],
    'key3': [('pattern3_1', 'value1')],
}

mapper_2 = {
    'key1': [('pattern1_1', 'value1'), ('pattern1_2', 'value2')],
    'key3': [('pattern3_1', 'value1'), ('pattern3_2', 'value2')],
}

dict_1 = {
    'key1': 'pattern1_2',
    'key2': 'pattern2_1',
    'key3': 'pattern3_3',
    'key4': 'pattern4_1'
}

dict_2 = {
    'key2': 'pattern2_2',
    'key3': 'pattern3_2',
    'key5': 'pattern5_1'
}

result_1_1 = {
    'key1': 'value2',
    'key2': 'value1',
    'key3': 'pattern3_3',
    'key4': 'pattern4_1'
}

result_2_1 = {
    'key1': 'value2',
    'key2': 'pattern2_1',
    'key3': 'pattern3_3',
    'key4': 'pattern4_1'
}

result_1_2 = {
    'key2': 'pattern2_2',
    'key3': 'pattern3_2',
    'key5': 'pattern5_1'
}

result_2_2 = {
    'key2': 'pattern2_2',
    'key3': 'value2',
    'key5': 'pattern5_1'
}


class TestMatchMap(unittest.TestCase):
    def test_dict_match_map(self):
        result = replace_attr_if_match(mapper_1, dict_1)
        self.assertEqual(result, result_1_1)
        result = replace_attr_if_match(mapper_2, dict_1)
        self.assertEqual(result, result_2_1)
        result = replace_attr_if_match(mapper_1, dict_2)
        self.assertEqual(result, result_1_2)
        result = replace_attr_if_match(mapper_2, dict_2)
        self.assertEqual(result, result_2_2)