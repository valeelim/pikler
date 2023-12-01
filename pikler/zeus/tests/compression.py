from django.test import TestCase
from zeus.compression import StandardPostings
from zeus.compression import VBEPostings

class UtilTestCase(TestCase):
    def setUp(self):
        self.postings_list = [34, 67, 89, 454, 2345738]
        self.tf_list = [12, 10, 3, 4, 1]

    def test_encode_decode(self):
        for Postings in [StandardPostings, VBEPostings]:
            print(Postings.__name__)
            encoded_postings_list = Postings.encode(self.postings_list)
            encoded_tf_list = Postings.encode_tf(self.tf_list)
            print("byte hasil encode postings: ", encoded_postings_list)
            print("ukuran encoded postings   : ",
                len(encoded_postings_list), "bytes")
            print("byte hasil encode TF list : ", encoded_tf_list)
            print("ukuran encoded postings   : ", len(encoded_tf_list), "bytes")

            decoded_posting_list = Postings.decode(encoded_postings_list)
            decoded_tf_list = Postings.decode_tf(encoded_tf_list)
            print("hasil decoding (postings): ", decoded_posting_list)
            print("hasil decoding (TF list) : ", decoded_tf_list)
            self.assertEqual(decoded_posting_list, self.postings_list, "hasil decoding tidak sama dengan postings original")
            self.assertEqual(decoded_tf_list, self.tf_list, "hasil decoding tidak sama dengan postings original")
            print()
