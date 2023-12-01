from django.core.management.base import BaseCommand
from tqdm import tqdm
import random
import os
import pandas as pd
from zeus.resources import RESOURCES_DIR

class Command(BaseCommand):
    help = 'Sample collections folder.'

    def handle(self, *args, **options):
        relevant_doc_ids = set()
        train_relevant_doc_ids = set()
        test_relevant_doc_ids = set()

        with open(RESOURCES_DIR / 'wikIR59k' / 'training' / 'qrels.txt', 'r') as f:
            for line in f:
                doc_id = line.split()[2]
                relevant_doc_ids.add(str(doc_id))
                train_relevant_doc_ids.add(str(doc_id))

        with open(RESOURCES_DIR / 'wikIR59k' / 'test' / 'qrels.txt', 'r') as f:
            for line in f:
                doc_id = line.split()[2]
                relevant_doc_ids.add(str(doc_id))
                test_relevant_doc_ids.add(str(doc_id))
        
        print(len(relevant_doc_ids))
        
        # relevant_doc_ids ~ 50% of docs
        # sample 100_000
        doc_path = RESOURCES_DIR / 'wikIR59k' / 'documents.csv'
        
        chunk_size = 10_000
        sample_size = 100_000
        prob_taken_not_relevant = .2
        sampled_docs = []

        for chunk in pd.read_csv(doc_path, chunksize=chunk_size, header=None):
            if len(sampled_docs) >= sample_size:
                break
            
            for _, row in tqdm(chunk.iterrows()):
                doc_id = str(row.iloc[0])
                if doc_id in relevant_doc_ids:
                    sampled_docs.append(row)
                elif random.random() < prob_taken_not_relevant:
                    sampled_docs.append(row)

                if len(sampled_docs) >= sample_size:
                    break
        
        # sampled_df = pd.DataFrame(sampled_docs)
        # sampled_df.to_csv(RESOURCES_DIR / 'wikIR59k' / 'training' / 'sampled_docs.csv', index=False)

        print('Train relevant count:\t', len([doc[0] for doc in sampled_docs if doc[0] in train_relevant_doc_ids]))
        print('Test relevant count:\t', len([doc[0] for doc in sampled_docs if doc[0] in test_relevant_doc_ids]))
