from django.urls import path
from .views import QueryViewSet
from .views import doc_details
from rest_framework.routers import SimpleRouter

router = SimpleRouter()
router.register(r'', QueryViewSet, basename='search')

urlpatterns = [path('doc/<str:folder>/<str:path>', doc_details, name='doc_details')]
urlpatterns += router.urls