from typing import Any
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from django.http import JsonResponse
from zeus.bsbi import BSBIIndex
from zeus.resources import RESOURCES_DIR
from zeus.compression import VBEPostings
from . import lm

from rest_framework.serializers import Serializer
from rest_framework.serializers import CharField
from rest_framework.serializers import IntegerField
from rest_framework.serializers import FloatField
from autocorrect import Speller

import os
import re

spell = Speller(lang='en')

class QuerySerializer(Serializer):
    doc_id = IntegerField()
    doc_path = CharField()
    title = CharField(max_length=255)
    score = FloatField()
    content = CharField()
    preview = CharField()


class QueryViewSet(viewsets.ViewSet):

    def __init__(self, **kwargs: Any) -> None:
        self.BSBI_instance = BSBIIndex(
            data_dir=(RESOURCES_DIR / 'collections'),
            postings_encoding=VBEPostings,
            output_dir=(RESOURCES_DIR / 'index'),
        )
        self.BSBI_instance.load()

        super().__init__(**kwargs)
    
    def list(self, request):
        query = request.GET.get('q', None) 
        page = int(request.GET.get('page', 1))

        if query is None:
            return Response([])
        
        context = {}
        spell_check = None
        
        if query != spell(query):
            spell_check = spell(query)
        
        bm25_result = self.BSBI_instance.retrieve_bm25(query, page * 10)
        letor_result = lm.evaluate_letor(query, [doc for _, doc in bm25_result])
        
        data = []
        for (score, doc) in letor_result[page * 10 - 10:]:
            did = int(os.path.splitext(os.path.basename(doc))[0])
            path_parts = doc.split(os.sep)
            collection_path = os.path.join(path_parts[-2], path_parts[-1])
            with open(doc, 'r') as f:
                content = f.readline()
                title, content = re.split(r'\s{4,}', content, 1)

                data.append({
                    'doc_id': did,
                    'doc_path': collection_path,
                    'title': title,
                    'score': score,
                    'content': content,
                    'preview': (content[:200] + '...') if len(content) > 200 else content,
                })
        
        context['data'] = data
        context['metadata'] = {
            'query': query,
            'page': page,
            'search_instead': spell_check,
        }

        return render(request, 'index.html', context)


def doc_details(request, folder: str, path: str):
    doc_path = os.path.join(folder, path)
    full_path = RESOURCES_DIR / 'collections' / f'{doc_path}'
    with open(full_path, 'r') as f:
        content = f.readline()
        title, content = re.split(r'\s{4,}', content, 1)

        context = {
            'title': title,
            'doc_path': doc_path,
            'content': content,
        }
        return render(request, 'page_detail.html', context)
