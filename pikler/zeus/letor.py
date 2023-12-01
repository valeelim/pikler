import os
import pandas as pd
import random
import pickle

import lightgbm as lgb
import numpy as np
from scipy.spatial.distance import cosine
from zeus.utils import PreProcessText
from zeus.utils import Singleton
from zeus.resources import RESOURCES_DIR
from tqdm import tqdm

class LambdaMart:
    __metaclass__ = Singleton
    NUM_NEGATIVES = 1
    NUM_LATENT_TOPICS = 200

    def __init__(self) -> None:
        from gensim.corpora import Dictionary

        self.documents = {}
        self.queries = {}
        self.val_queries = {}

        self.q_docs_rel = {}
        self.val_q_docs_rel = {}

        self.group_qid_count = []
        self.val_group_qid_count = []

        self.dataset = []
        self.val_dataset = []
    
        self.preprocess_text = PreProcessText()

        self.dictionary = Dictionary()
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            boosting_type = "gbdt",
            n_estimators = 600,
            importance_type = "gain",
            metric = "ndcg",
            num_leaves = 60,
            learning_rate = 0.01,
            max_depth = -1,
        )

        ranker_path = RESOURCES_DIR / 'tmp' / 'ranker.pkl'
        model_path = RESOURCES_DIR / 'tmp' / 'lsi_model.pkl'
        if os.path.exists(ranker_path) and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(ranker_path, 'rb') as f:
                self.ranker = pickle.load(f)
            print('Ranker loaded...')

        else:
            print('Begin training...')
            self.load_documents(RESOURCES_DIR / 'wikiclir' / 'wikiclir_docs.txt')
            print('Loading Queries...')
            self.load_queries(RESOURCES_DIR / 'wikiclir' / 'train_queries.txt')
            print('Loading Qrels...')
            self.load_qrels(RESOURCES_DIR / 'wikiclir' / 'wikiclear_qrels.txt')
            print('Constructing Dataset...')
            self.construct_dataset()

            print('Building LSI Model...')
            self.build_lsi_model()
            print('Fitting Dataset...')
            self.fit_dataset()
    
    def load_documents(self, train_docs_path: str) -> None:
        tmp_path = RESOURCES_DIR / 'tmp' / 'documents.pkl'
        if os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                self.documents = pickle.load(f)
            return

        with open(train_docs_path, 'r') as f:
            lines = f.readlines()

            for line in tqdm(lines):
                doc_id, content = line.strip().split(' ', 1)
                self.documents[doc_id] = self.preprocess_text.run(content)
    
        assert len(self.documents) == len(lines), 'Number of documents does not match.'
        with open(tmp_path, 'wb+') as f:
            pickle.dump(self.documents, f)
    
    def load_queries(self, train_queries_path: str) -> None:
        tmp_path = RESOURCES_DIR / 'tmp' / 'train_queries.pkl'
        if os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                self.queries = pickle.load(f)
            return
        
        with open(train_queries_path, 'r') as f:
            lines = f.readlines()

            for line in tqdm(lines):
                q_id, content = line.strip().split(' ', 1)
                self.queries[q_id] = self.preprocess_text.run(content)
        
        assert len(self.queries) == len(lines), 'Number of queries does not match.'
        with open(tmp_path, 'wb+') as f:
            pickle.dump(self.queries, f)
    
    def load_val_queries(self, val_queries_path: str) -> None:
        tmp_path = RESOURCES_DIR / 'tmp' / 'val_queries.pkl'
        if os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                self.val_queries = pickle.load(f)
            return

        df = pd.read_csv(val_queries_path, header=None, names=['doc_id', 'content'])
        for _, row in tqdm(df.iterrows()):
            self.val_queries[row['doc_id']] = self.preprocess_text.run((row['content']))

        assert len(self.val_queries) == len(df), 'Number of documents does not match.'
        with open(tmp_path, 'wb+') as f:
            pickle.dump(self.val_queries, f)
    
    def load_qrels(self, train_qrel_path: str) -> None:
        tmp_path = RESOURCES_DIR / 'tmp' / 'q_docs_rel.pkl'
        if os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                self.q_docs_rel = pickle.load(f)
            return

        with open(train_qrel_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                q_id, doc_id, rel = line.strip().split()
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in self.q_docs_rel:
                        self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))

        with open(tmp_path, 'wb+') as f:
            pickle.dump(self.q_docs_rel, f)

    def load_val_qrels(self, val_qrels_path: str) -> None:
        tmp_path = RESOURCES_DIR / 'tmp' / 'val_q_docs_rel.pkl'
        if os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                self.val_q_docs_rel = pickle.load(f)
            return

        with open(val_qrels_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                q_id, _, doc_id, rel = line.strip().split()
                q_id = int(q_id)
                doc_id = int(doc_id)
                rel = int(rel)
                if (q_id in self.val_queries) and (doc_id in self.documents):
                    if q_id not in self.val_q_docs_rel:
                        self.val_q_docs_rel[q_id] = []
                    self.val_q_docs_rel[q_id].append((doc_id, int(rel)))

        with open(tmp_path, 'wb+') as f:
            pickle.dump(self.val_q_docs_rel, f)
        
    def construct_dataset(self) -> None:
        tmp_path = RESOURCES_DIR / 'tmp' / 'dataset.pkl'
        tmp_path_2 = RESOURCES_DIR / 'tmp' / 'group_qid_count.pkl'
        if os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                self.dataset = pickle.load(f)

            if os.path.exists(tmp_path_2):
                with open(tmp_path_2, 'rb') as f:
                    self.group_qid_count = pickle.load(f)
            return

        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + LambdaMart.NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.documents[doc_id], rel))

            for _ in range(LambdaMart.NUM_NEGATIVES):
                self.dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))

        assert sum(self.group_qid_count) == len(self.dataset)
        print(len(self.dataset))
        with open(tmp_path, 'wb+') as f:
            pickle.dump(self.dataset, f)
        with open(tmp_path_2, 'wb+') as f:
            pickle.dump(self.group_qid_count, f)
    
    def construct_val_dataset(self) -> None:
        tmp_path = RESOURCES_DIR / 'tmp' / 'val_dataset.pkl'
        tmp_path_2 = RESOURCES_DIR / 'tmp' / 'val_group_qid_count.pkl'
        if os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                self.val_dataset = pickle.load(f)

            if os.path.exists(tmp_path_2):
                with open(tmp_path_2, 'rb') as f:
                    self.val_group_qid_count = pickle.load(f)
            return

        for q_id in self.val_q_docs_rel:
            docs_rels = self.val_q_docs_rel[q_id]
            self.val_group_qid_count.append(len(docs_rels) + LambdaMart.NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.val_dataset.append((self.val_queries[q_id], self.documents[doc_id], rel))
            
            for _ in range(LambdaMart.NUM_NEGATIVES):
                self.val_dataset.append((self.val_queries[q_id], random.choice(list(self.documents.values())), 0))
        
        assert sum(self.val_group_qid_count) == len(self.val_dataset)
        with open(tmp_path, 'wb+') as f:
            pickle.dump(self.val_dataset, f)
        with open(tmp_path_2, 'wb+') as f:
            pickle.dump(self.val_group_qid_count, f)
    
    def build_lsi_model(self) -> None:
        from gensim.models import LsiModel
        tmp_path = RESOURCES_DIR / 'tmp' / 'lsi_model.pkl'
        if os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                self.model = pickle.load(f)
            return

        bow_corpus = [self.dictionary.doc2bow(doc, allow_update=True) for doc in self.documents.values()]
        self.model = LsiModel(bow_corpus, num_topics=LambdaMart.NUM_LATENT_TOPICS)
        with open(tmp_path, 'wb+') as f:
            pickle.dump(self.model, f)


    def _vector_rep(self, text: str):
        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == LambdaMart.NUM_LATENT_TOPICS else [0.] * LambdaMart.NUM_LATENT_TOPICS

    def features(self, query: list[str], doc: list[str]) -> float:

        v_q = self._vector_rep(query)
        v_d = self._vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)

        return v_q + v_d + [jaccard] + [cosine_dist]
    
    def fit_dataset(self) -> None:
        tmp_path = RESOURCES_DIR / 'tmp' / 'ranker.pkl'
        if os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                self.ranker = pickle.load(f)
            return

        X = []
        Y = []
        for (query, doc, rel) in self.dataset:
            X.append(self.features(query, doc))
            Y.append(rel)
        
        X = np.array(X)
        Y = np.array(Y)

        # validation
        X_val = []
        Y_val = []
        for (query, doc, rel) in self.val_dataset:
            X_val.append(self.features(query, doc))
            Y_val.append(rel)
        
        X_val = np.array(X_val)
        Y_val = np.array(Y_val)

        self.ranker.fit(X, Y, group=self.group_qid_count)

        with open(tmp_path, 'wb+') as f:
            pickle.dump(self.ranker, f)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.ranker.predict(X)
    
    def evaluate_letor(self, query: str, doc_path: list[str]) -> list[tuple[float, str]]:
        if not doc_path:
            return []

        X = []
        for doc in doc_path:
            with open(doc, 'r') as f:
                X.append(self.features(self.preprocess_text.run((query)), self.preprocess_text.run((f.readline()))))

        X = np.array(X)
        scores = self.predict(X)

        did_scores = [x for x in zip(scores, doc_path)]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[0], reverse = True)

        return sorted_did_scores
