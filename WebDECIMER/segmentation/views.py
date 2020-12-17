from django.shortcuts import render

from rest_framework import viewsets
from .serializers import UploadedArticleSerializer
from .serializers import SegmentedImageSerializer

from .models import UploadedArticle
from .models import SegmentedImage

from django_filters.rest_framework import DjangoFilterBackend



class UploadedArticleViewSet(viewsets.ModelViewSet):
    queryset = UploadedArticle.objects.all().order_by('-uploaded')
    serializer_class = UploadedArticleSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['ori_name']

class SegmentedImageViewSet(viewsets.ModelViewSet):
    queryset = SegmentedImage.objects.all()
    serializer_class = SegmentedImageSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['ori_article_id', 'smiles']

