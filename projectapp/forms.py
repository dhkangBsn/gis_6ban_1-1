from django.forms import ModelForm
from django import forms
from projectapp.models import Project


class ProjectCreationForm(ModelForm):
    is_private = forms.BooleanField(required=False)
    class Meta:
        model = Project
        fields = ['title','image','description','is_private']
