from django.test import TestCase
from zeus.compression import VBEPostings
from zeus.index import InvertedIndexWriter
from . import TEST_DIR

class IndexTestCase(TestCase):
    def test_index(self):
        with InvertedIndexWriter('test', postings_encoding=VBEPostings, directory=(TEST_DIR / 'tmp')) as index:
            index.append(1, [2, 3, 4, 8, 10], [2, 4, 2, 3, 30])
            index.append(2, [3, 4, 5], [34, 23, 56])
            index.index_file.seek(0)
            self.assertEqual(index.terms, [1, 2], "terms salah")
            self.assertEqual(index.doc_length, {2: 2, 3: 38, 4: 25, 5: 56, 8: 3, 10: 30}, "doc_length salah")
            self.assertEqual(index.postings_dict, {1: (0,5,len(VBEPostings.encode([2, 3, 4, 8, 10])), len(VBEPostings.encode_tf([2, 4, 2, 3, 30]))), 2: (len(VBEPostings.encode([2, 3, 4, 8, 10])) + len(VBEPostings.encode_tf([2, 4, 2, 3, 30])), 3, len(VBEPostings.encode([3, 4, 5])), len(VBEPostings.encode_tf([34, 23, 56])))}, "postings dictionary salah")
            index.index_file.seek(index.postings_dict[2][0])
            self.assertEqual(VBEPostings.decode(index.index_file.read(len(VBEPostings.encode([3, 4, 5])))), [3, 4, 5], "terdapat kesalahan")
            self.assertEqual(VBEPostings.decode_tf(index.index_file.read(len(VBEPostings.encode_tf([34, 23, 56])))), [34, 23, 56], "terdapat kesalahan")
