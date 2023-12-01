from django.core.management.base import BaseCommand
import os
import pandas as pd
from tqdm import tqdm
from zeus.resources import RESOURCES_DIR

class Command(BaseCommand):
    help = 'Construct collections folder.'

    def handle(self, *args, **options):
        files_per_folder = 10000
        folder_counter = 0
        file_counter = 0

        with open(RESOURCES_DIR / 'wikiclir' / 'wikiclir_docs.txt', 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                doc_id = parts[0]
                text = parts[1]

                if file_counter % files_per_folder == 0:
                    folder_counter += 1
                    current_folder_path = os.path.join(RESOURCES_DIR / 'collections' / f'{folder_counter}')
                    os.makedirs(current_folder_path, exist_ok=True)

                file_path = os.path.join(current_folder_path, f'{doc_id}.txt')

                with open(file_path, 'w') as text_file:
                    text_file.write(text)

                file_counter += 1
                
        print('All documents processed.')
