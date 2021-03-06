from django.contrib.auth.decorators import login_required
from django.db.models import Avg
from django.shortcuts import render

# Create your views here.
from django.urls import reverse_lazy, reverse
from django.utils.decorators import method_decorator
from django.views.generic import CreateView, DetailView, ListView, UpdateView
from django.views.generic.edit import FormMixin, DeleteView
from django.views.generic.list import MultipleObjectMixin

from articleapp.models import Article
from emotionapp.decorators import emotion_ownership_required
from emotionapp.forms import EmotionCreationForm
from emotionapp.models import Emotion
from projectapp.models import Project
from subscribeapp.models import Subscription


@method_decorator(login_required, 'get')
@method_decorator(login_required, 'post')
class EmotionCreateView(CreateView):
    model = Emotion
    form_class = EmotionCreationForm
    # success_url = reverse_lazy('articleapp:list') login_required를 적용하였으므로 아래의와 같이 get_success_url 메소드를 재정의한다.
    template_name = 'emotionapp/create.html'
    def form_valid(self, form):
        form.instance.writer = self.request.user #request를 보내는 user로 writer를 할당
        return super().form_valid(form)
    def get_success_url(self):
        return reverse("emotionapp:detail" , kwargs={"pk":self.object.pk})


class EmotionListView(ListView):
    model = Emotion
    context_object_name = 'emotion_list'
    template_name = 'emotionapp/list.html'
    paginate_by = 20
    def get_context_data(self, **kwargs):

        lat_lon_joy = Article.objects.filter(joy__gt=60).values('lat','lon') # 기쁨
        lat_lon_sadness = Article.objects.filter(sadness__gt=60).values('lat','lon') # 슬픔
        lat_lon_fear = Article.objects.filter(fear__gt=60).values('lat','lon') # 놀람
        lat_lon_surprise = Article.objects.filter(surprise__gt=60).values('lat','lon') # 상처
        lat_lon_anger = Article.objects.filter(anger__gt=60).values('lat','lon') # 분노
        lat_lon_love = Article.objects.filter(love__gt=60).values('lat','lon') # 두려움

        lat_lon_1 = lat_lon_joy.union(lat_lon_sadness)
        print(lat_lon_1)
        lat_lon_2 = lat_lon_fear.union(lat_lon_surprise)
        print(lat_lon_2)
        lat_lon_3 = lat_lon_anger.union(lat_lon_love)
        print(lat_lon_3)
        lat_lon_4 = lat_lon_1.union(lat_lon_2)

        lat_lon_5 = lat_lon_4.union(lat_lon_3)

        joy = Article.objects.aggregate(Avg('joy')) # 기쁨
        joy['joy__avg'] = round(joy['joy__avg'], 2)
        print(joy)
        sadness = Article.objects.aggregate(Avg('sadness'))  # 기쁨
        sadness['sadness__avg']=round(sadness['sadness__avg'],2)

        fear = Article.objects.aggregate(Avg('fear'))  # 기쁨
        fear['fear__avg'] = round(fear['fear__avg'], 2)

        surprise = Article.objects.aggregate(Avg('surprise'))  # 기쁨
        surprise['surprise__avg'] = round(surprise['surprise__avg'], 2)

        anger = Article.objects.aggregate(Avg('anger'))  # 기쁨
        anger['anger__avg'] = round(anger['anger__avg'], 2)

        love = Article.objects.aggregate(Avg('love'))  # 기쁨
        love['love__avg'] = round(love['love__avg'], 2)



        return super().get_context_data(lat_lon_joy=lat_lon_joy,lat_lon_sadness=lat_lon_sadness,
                                        lat_lon_fear=lat_lon_fear,lat_lon_surprise=lat_lon_surprise,
                                        lat_lon_anger=lat_lon_anger,lat_lon_love=lat_lon_love,lat_lon=lat_lon_5,
                                        joy=joy,sadness=sadness,fear=fear,surprise=surprise,anger=anger,love=love,
                                        object_name='article',**kwargs)


@method_decorator(emotion_ownership_required,'get')
@method_decorator(emotion_ownership_required,'post')
class EmotionUpdateView(UpdateView):
    model = Emotion
    form_class = EmotionCreationForm
    context_object_name = 'target_emotion'
    template_name = 'emotionapp/update.html'
    def get_success_url(self):
        return reverse('emotionapp/detail_article.html', kwargs={'pk':self.object.pk})

@method_decorator(emotion_ownership_required,'get')
@method_decorator(emotion_ownership_required,'post')
class EmotionDeleteView(DeleteView):
    model = Emotion
    context_object_name = 'target_emotion'
    success_url = reverse_lazy('emotionapp:list')
    template_name = 'emotionapp/delete.html'

class EmotionDetailView(DetailView):
    model = Emotion
    context_object_name = 'target_emotion'
    template_name = 'emotionapp/detail.html'
    paginate_by = 20

    def get_context_data(self, **kwargs):
        emotion = self.object.emotion
        article_list = None
        if emotion == 'joy':article_list = Article.objects.filter(joy__gt=70) # 기쁨
        elif emotion == 'sadness':article_list = Article.objects.filter(sadness__gt=70) # 슬픔
        elif emotion == 'fear':article_list = Article.objects.filter(fear__gt=70) # 놀람
        elif emotion == 'surprise':article_list = Article.objects.filter(surprise__gt=70) # 상처
        elif emotion == 'anger':article_list = Article.objects.filter(anger__gt=70) # 분노
        elif emotion == 'love':article_list = Article.objects.filter(love__gt=70) # 두려움
        lat_lon = article_list.values('lat','lon')
        return super().get_context_data(object_list=article_list,lat_lon=lat_lon,object_name='article',**kwargs)
