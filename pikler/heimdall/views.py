from typing import Any
from rest_framework import viewsets
from rest_framework.response import Response
from zeus.bsbi import BSBIIndex
from zeus.resources import RESOURCES_DIR
from zeus.compression import VBEPostings
from . import lm

from rest_framework.serializers import Serializer
from rest_framework.serializers import CharField
from rest_framework.serializers import IntegerField
from rest_framework.serializers import FloatField

import os
import re

class QuerySerializer(Serializer):
    doc_id = IntegerField()
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

        bm25_result = self.BSBI_instance.retrieve_bm25(query, page * 10)
        letor_result = lm.evaluate_letor(query, [doc for _, doc in bm25_result])
        
        data = []
        for (score, doc) in letor_result[page * 10 - 10:]:
            did = int(os.path.splitext(os.path.basename(doc))[0])
            with open(doc, 'r') as f:
                content = f.readline()
                title, content = re.split(r'\s{4,}', content, 1)

                data.append({
                    'doc_id': did,
                    'title': title,
                    'score': score,
                    'content': content,
                    'preview': (content[:75] + '...') if len(content) > 75 else content,
                })
        
        results = QuerySerializer(data, many=True)
        return Response(results.data)
        