import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer



class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        return len(self.str_to_id)

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        if s not in self:
            self.str_to_id[s] = len(self.str_to_id)
            self.id_to_str.append(s)
            
        return self.str_to_id[s]

    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        return self.id_to_str[i]

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        Jika key adalah integer, gunakan __get_str;
        jika key adalah string, gunakan __get_id
        """
        if isinstance(key, str):
            return self.__get_id(key)

        elif isinstance(key, int):
            return self.__get_str(key)

        return None
    
    def __contains__(self, query):
        return query in self.str_to_id


def merge_and_sort_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabung (merge) dua lists of tuples (doc id, tf) dan mengembalikan
    hasil penggabungan keduanya (TF perlu diakumulasikan untuk semua tuple
    dengn doc id yang sama), dengan aturan berikut:

    contoh: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Dua buah sorted list of tuples yang akan di-merge.

    Returns
    -------
    List[(Comparable, int)]
        Penggabungan yang sudah terurut
    """
    tf1, tf2 = 0, 0
    result = []
    while tf1 < len(posts_tfs1) and tf2 < len(posts_tfs2):
        if posts_tfs1[tf1][0] == posts_tfs2[tf2][0]:
            result.append((posts_tfs1[tf1][0], posts_tfs1[tf1][1] + posts_tfs2[tf2][1]))
            tf1 += 1
            tf2 += 1
        
        elif posts_tfs1[tf1][0] < posts_tfs2[tf2][0]:
            result.append(posts_tfs1[tf1])
            tf1 += 1
        
        else:
            result.append(posts_tfs2[tf2])
            tf2 += 1

    while tf1 < len(posts_tfs1):
        result.append(posts_tfs1[tf1])
        tf1 += 1
    
    while tf2 < len(posts_tfs2):
        result.append(posts_tfs2[tf2])
        tf2 += 1
    
    return result

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class PreProcessText(object):
    __metaclass__ = Singleton
    nltk.download('punkt')
    nltk.download('stopwords')

    def __init__(self, *args, **kwargs) -> None:
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))  # Load stopwords once
        super().__init__(*args, **kwargs)

    def run(self, text: str) -> list[str]:
        tokens = re.findall('\w+', text)
        processed_tokens = [self.stemmer.stem(word) for word in tokens if word.lower() not in self.stop_words and word not in string.punctuation]

        return processed_tokens

