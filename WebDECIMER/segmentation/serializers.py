from .models import SegmentedImage
from .models import UploadedArticle

from rest_framework import serializers

class UploadedArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedArticle
        fields = '__all__'

class SegmentedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = SegmentedImage
        fields = '__all__'

        