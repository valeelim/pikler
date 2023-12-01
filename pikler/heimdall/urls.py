from django.urls import path
from .views import QueryViewSet
from rest_framework.routers import SimpleRouter

router = SimpleRouter()
router.register(r'search', QueryViewSet, basename='search')

urlpatterns = router.urls