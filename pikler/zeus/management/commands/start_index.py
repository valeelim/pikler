from django.core.management.base import BaseCommand
from zeus.compression import VBEPostings
from zeus.resources import RESOURCES_DIR
from zeus.bsbi import BSBIIndex

class Command(BaseCommand):
    help = 'Start indexing for a given collection'

    def handle(self, *args, **options):
        BSBI_instance = BSBIIndex(
            data_dir=(RESOURCES_DIR / 'collections'),
            postings_encoding=VBEPostings,
            output_dir=(RESOURCES_DIR / 'index'),
        )

        BSBI_instance.do_indexing()