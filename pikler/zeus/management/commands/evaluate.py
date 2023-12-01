import os
from zeus.letor import LambdaMart
from django.core.management.base import BaseCommand
import pandas as pd
import math
from zeus.bsbi import BSBIIndex
from zeus.compression import VBEPostings
from collections import defaultdict
from tqdm import tqdm
from zeus.resources import RESOURCES_DIR


class Command(BaseCommand):
    help = 'Evaluate search effectiveness'

    def rbp(self, ranking, p=0.8):
        """ menghitung search effectiveness metric score dengan 
            Rank Biased Precision (RBP)

            Parameters
            ----------
            ranking: List[int]
            vektor biner seperti [1, 0, 1, 1, 1, 0]
            gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
            Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                    di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                    di rank-6 tidak relevan

            Returns
            -------
            Float
            score RBP
        """
        score = 0.
        for i in range(1, len(ranking) + 1):
            pos = i - 1
            score += ranking[pos] * (p ** (i - 1))
        return (1 - p) * score


    def dcg(self, ranking):
        """ menghitung search effectiveness metric score dengan 
            Discounted Cumulative Gain

            Parameters
            ----------
            ranking: List[int]
            vektor biner seperti [1, 0, 1, 1, 1, 0]
            gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
            Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                    di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                    di rank-6 tidak relevan

            Returns
            -------
            Float
            score DCG
        """
        dcg_value = 0
        for i, rel in enumerate(ranking, 1):
            dcg_value += rel / (math.log2(i + 1))
        return dcg_value

    def idcg(self, ranking):
        ideal_ranking = sorted(ranking, reverse=True)
        return self.dcg(ideal_ranking)

    def ndcg(self, ranking):
        c_dcg = self.dcg(ranking)
        c_idcg = self.idcg(ranking)
        return c_dcg / c_idcg if c_idcg > 0 else 0


    def prec(self, ranking, k):
        """ menghitung search effectiveness metric score dengan 
            Precision at K

            Parameters
            ----------
            ranking: List[int]
            vektor biner seperti [1, 0, 1, 1, 1, 0]
            gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
            Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                    di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                    di rank-6 tidak relevan

            k: int
            banyak dokumen yang dipertimbangkan atau diperoleh

            Returns
            -------
            Float
            score Prec@K
        """
        k = min(k, len(ranking))
        top_k_relevant = sum(ranking[:k])
        return top_k_relevant / k


    def ap(self, ranking):
        """ menghitung search effectiveness metric score dengan 
            Average Precision

            Parameters
            ----------
            ranking: List[int]
            vektor biner seperti [1, 0, 1, 1, 1, 0]
            gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
            Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                    di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                    di rank-6 tidak relevan

            Returns
            -------
            Float
            score AP
        """
        total_relevant = sum(ranking)
        if total_relevant == 0:
            return 0.0
        
        sum_precision = 0.0
        count_relevant = 0
        
        for k, rel in enumerate(ranking, 1):
            if rel == 1:
                count_relevant += 1
                precision_at_k = count_relevant / k
                sum_precision += precision_at_k

        return sum_precision / total_relevant

    # >>>>> memuat qrels


    def load_qrels(self, qrel_file="qrels.txt"):
        """ 
            memuat query relevance judgment (qrels) 
            dalam format dictionary of dictionary qrels[query id][document id],
            dimana hanya dokumen yang relevan (nilai 1) yang disimpan,
            sementara dokumen yang tidak relevan (nilai 0) tidak perlu disimpan,
            misal {"Q1": {500:1, 502:1}, "Q2": {150:1}}
        """
        qrels = defaultdict(lambda: defaultdict(lambda: 0)) 
        with open(qrel_file) as file:
            for line in file:
                parts = line.strip().split()
                qid = parts[0]
                did = int(parts[1])
                qrels[qid][did] = 1
        return qrels

    # >>>>> EVALUASI !


    def eval_retrieval(self, qrels, query_file="queries.txt", k=100):
        """ 
        loop ke semua query, hitung score di setiap query,
        lalu hitung MEAN SCORE-nya.
        untuk setiap query, kembalikan top-100 documents
        """
        BSBI_instance = BSBIIndex(
            data_dir=(RESOURCES_DIR / 'collections'),
            postings_encoding=VBEPostings,
            output_dir=(RESOURCES_DIR / 'index'),
        )
        
        BSBI_instance.load()

        lm = LambdaMart()

        with open(query_file) as file:
            rbp_scores_tfidf = []
            dcg_scores_tfidf = []
            ndcg_scores_tfidf = []
            ap_scores_tfidf = []

            rbp_scores_bm25 = []
            dcg_scores_bm25 = []
            ndcg_scores_bm25 = []
            ap_scores_bm25 = []

            rbp_scores_bm25_letor = []
            dcg_scores_bm25_letor = []
            ndcg_scores_bm25_letor = []
            ap_scores_bm25_letor = []

            rbp_scores_tfidf_letor = []
            dcg_scores_tfidf_letor = []
            ndcg_scores_tfidf_letor = []
            ap_scores_tfidf_letor = []

            for qline in tqdm(file):
                parts = qline.strip().split()
                qid = parts[0]
                query = " ".join(parts[1:])

                """
                Evaluasi TF-IDF
                """
                ranking_tfidf = []
                tfidf_result = BSBI_instance.retrieve_tfidf(query, k=k) # (score, doc)
                for (score, doc) in tfidf_result:
                    did = int(os.path.splitext(os.path.basename(doc))[0])
                    # Alternatif lain:
                    # 1. did = int(doc.split("\\")[-1].split(".")[0])
                    # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                    # 3. disesuaikan dengan path Anda
                    if (did in qrels[qid]):
                        ranking_tfidf.append(1)
                    else:
                        ranking_tfidf.append(0)
                rbp_scores_tfidf.append(self.rbp(ranking_tfidf))
                dcg_scores_tfidf.append(self.dcg(ranking_tfidf))
                ndcg_scores_tfidf.append(self.ndcg(ranking_tfidf))
                ap_scores_tfidf.append(self.ap(ranking_tfidf))

                """
                Evaluasi BM25
                """
                ranking_bm25 = []
                bm25_result = BSBI_instance.retrieve_bm25(query, k=k)
                # nilai k1 dan b dapat diganti-ganti
                for (score, doc) in bm25_result:
                    did = int(os.path.splitext(os.path.basename(doc))[0])
                    # Alternatif lain:
                    # 1. did = int(doc.split("\\")[-1].split(".")[0])
                    # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                    # 3. disesuaikan dengan path Anda
                    if (did in qrels[qid]):
                        ranking_bm25.append(1)
                    else:
                        ranking_bm25.append(0)

                rbp_scores_bm25.append(self.rbp(ranking_bm25))
                dcg_scores_bm25.append(self.dcg(ranking_bm25))
                ndcg_scores_bm25.append(self.ndcg(ranking_bm25))
                ap_scores_bm25.append(self.ap(ranking_bm25))

                ranking_letor_bm25 = []
                letor_bm25 = lm.evaluate_letor(query, [doc for _, doc in bm25_result])
                for (score, doc) in letor_bm25:
                    did = int(os.path.splitext(os.path.basename(doc))[0])
                    # Alternatif lain:
                    # 1. did = int(doc.split("\\")[-1].split(".")[0])
                    # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                    # 3. disesuaikan dengan path Anda
                    if (did in qrels[qid]):
                        ranking_letor_bm25.append(1)
                    else:
                        ranking_letor_bm25.append(0)

                rbp_scores_bm25_letor.append(self.rbp(ranking_letor_bm25))
                dcg_scores_bm25_letor.append(self.dcg(ranking_letor_bm25))
                ndcg_scores_bm25_letor.append(self.ndcg(ranking_letor_bm25))
                ap_scores_bm25_letor.append(self.ap(ranking_letor_bm25))

                ranking_letor_tfidf = []
                letor_tfidf = lm.evaluate_letor(query, [doc for _, doc in tfidf_result])
                for (score, doc) in letor_tfidf:
                    did = int(os.path.splitext(os.path.basename(doc))[0])
                    # Alternatif lain:
                    # 1. did = int(doc.split("\\")[-1].split(".")[0])
                    # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                    # 3. disesuaikan dengan path Anda
                    if (did in qrels[qid]):
                        ranking_letor_tfidf.append(1)
                    else:
                        ranking_letor_tfidf.append(0)

                rbp_scores_tfidf_letor.append(self.rbp(ranking_letor_tfidf))
                dcg_scores_tfidf_letor.append(self.dcg(ranking_letor_tfidf))
                ndcg_scores_tfidf_letor.append(self.ndcg(ranking_letor_tfidf))
                ap_scores_tfidf_letor.append(self.ap(ranking_letor_tfidf))

        print("Hasil evaluasi TF-IDF terhadap 150 queries")
        print("RBP score =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
        print("DCG score =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
        print("AP score  =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))
        print("NDCG score  =", sum(ndcg_scores_tfidf) / len(ndcg_scores_tfidf))

        print("Hasil evaluasi TFIDF-Letor terhadap 150 queries")
        print("RBP score =", sum(rbp_scores_tfidf_letor) / len(rbp_scores_tfidf_letor))
        print("DCG score =", sum(dcg_scores_tfidf_letor) / len(dcg_scores_tfidf_letor))
        print("AP score  =", sum(ap_scores_tfidf_letor) / len(ap_scores_tfidf_letor))
        print("NDCG score  =", sum(ndcg_scores_tfidf_letor) / len(ndcg_scores_tfidf_letor))

        print("Hasil evaluasi BM25 terhadap 150 queries")
        print("RBP score =", sum(rbp_scores_bm25) / len(rbp_scores_bm25))
        print("DCG score =", sum(dcg_scores_bm25) / len(dcg_scores_bm25))
        print("AP score  =", sum(ap_scores_bm25) / len(ap_scores_bm25))
        print("NDCG score  =", sum(ndcg_scores_bm25) / len(ndcg_scores_bm25))

        print("Hasil evaluasi BM25-Letor terhadap 150 queries")
        print("RBP score =", sum(rbp_scores_bm25_letor) / len(rbp_scores_bm25_letor))
        print("DCG score =", sum(dcg_scores_bm25_letor) / len(dcg_scores_bm25_letor))
        print("AP score  =", sum(ap_scores_bm25_letor) / len(ap_scores_bm25_letor))
        print("NDCG score  =", sum(ndcg_scores_bm25_letor) / len(ndcg_scores_bm25_letor))

        # print(len(bm25_result), len(tfidf_result), len(letor_bm25), len(letor_tfidf))


    def handle(self, *args, **options):
        qrels = self.load_qrels(RESOURCES_DIR / 'wikiclir' / 'wikiclir_qrels.txt')

        self.eval_retrieval(qrels, RESOURCES_DIR / 'wikiclir' / 'test_queries.txt')
