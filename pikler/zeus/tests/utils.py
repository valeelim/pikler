from django.test import TestCase
from zeus.utils import IdMap
from zeus.utils import merge_and_sort_posts_and_tfs

class UtilTestCase(TestCase):
    def setUp(self):
        self.doc = ["halo", "semua", "selamat", "pagi", "semua"]
        self.term_id_map = IdMap()

    def test_term_id_map(self):
        self.assertEqual([self.term_id_map[term]
            for term in self.doc], [0, 1, 2, 3, 1], "term_id salah")
        self.assertEqual(self.term_id_map[1], "semua", "term_id salah")
        self.assertEqual(self.term_id_map[0], "halo", "term_id salah")
        self.assertEqual(self.term_id_map["selamat"], 2, "term_id salah")
        self.assertEqual(self.term_id_map["pagi"], 3, "term_id salah")
    
    def test_doc_id_map(self):
        docs = ["/collection/0/data0.txt",
                "/collection/0/data10.txt",
                "/collection/1/data53.txt"]
        doc_id_map = IdMap()
        self.assertEqual([doc_id_map[docname]
            for docname in docs], [0, 1, 2], "docs_id salah")
        
    def test_merge_and_sort_posts_and_tfs(self):
        self.assertEqual(merge_and_sort_posts_and_tfs([(1, 34), (3, 2), (4, 23)],
            [(1, 2), (2, 3), (3, 4)]), [(1, 36), (2, 3), (3, 6), (4, 23)], "merge_and_sort_posts_and_tfs salah")
