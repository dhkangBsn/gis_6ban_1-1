from django.forms import ModelForm
from django import forms

from emotionapp.models import Emotion


class EmotionCreationForm(ModelForm):
    is_private = forms.BooleanField(required=False)
    class Meta:
        model = Emotion
        fields = ['title','emotion','image','description','is_private']
