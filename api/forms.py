from django import forms
from .models import AudioModel

class UploadForm(forms.ModelForm):
    class Meta:
        model = AudioModel
        fields = ('file',)