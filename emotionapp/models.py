from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Emotion(models.Model):
    title = models.CharField(max_length=20, null=False)
    description = models.CharField(max_length=200, null=True)
    image = models.ImageField(upload_to='emotion/', null=False)
    writer = models.ForeignKey(User, on_delete=models.SET_NULL,related_name='emotion', null=True)
    created_at = models.DateField(auto_now_add=True, null=True)
    is_private = models.BooleanField(default=False, null=False)
    emotion = models.CharField(max_length=20, null=False)

    def __str__(self):
        return f'{self.title}'
