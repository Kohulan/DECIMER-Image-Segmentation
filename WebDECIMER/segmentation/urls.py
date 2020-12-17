from .views import UploadedArticleViewSet, SegmentedImageViewSet
from rest_framework import routers
from django.urls import path, include

app_name = 'api-segmentation'

router = routers.DefaultRouter()

router.register(r'segmentation/uploaded', UploadedArticleViewSet)
router.register(r'segmentation/segmented', SegmentedImageViewSet)

urlpatterns = [
    path('', include(router.urls)),

]
