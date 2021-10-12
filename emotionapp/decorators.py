from django.http import HttpResponseForbidden

from emotionapp.models import Emotion


def emotion_ownership_required(func):
    def decorated(request, *args, **kwargs):
        target_emotion = Emotion.objects.get(pk=kwargs['pk'])
        if target_emotion.writer == request.user:
            return func(request, *args, **kwargs)
        else:
            return HttpResponseForbidden()
    return decorated