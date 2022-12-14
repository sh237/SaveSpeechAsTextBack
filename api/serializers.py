from rest_framework import serializers
from .models import AudioModel,TextModel
from django.core.validators import FileExtensionValidator

class AudioFileSerializer(serializers.Serializer):
    file = serializers.FileField(
        validators=[FileExtensionValidator(allowed_extensions=['wav', 'caf', 'mp3', 'm4a', 'flac'])])

    class Meta:
        model = AudioModel
        fields = ('file',)

    def create(self,validated_data):
        return AudioModel.objects.create(**validated_data)

#TextSerializer
class TextSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=1000)

    class Meta:
        model = TextModel
        fields = ('text',)

    def is_valid(self, *, raise_exception=False):
        return super().is_valid(raise_exception=raise_exception)

    def create(self,validated_data):
        return TextModel.objects.create(**validated_data)