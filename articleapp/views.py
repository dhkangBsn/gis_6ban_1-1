from django.contrib.auth.decorators import login_required

# Create your views here.
from django.urls import reverse_lazy, reverse
from django.utils.decorators import method_decorator
from django.views.generic import CreateView, DetailView, UpdateView, DeleteView, ListView
from django.views.generic.edit import FormMixin

from articleapp.decorators import article_ownership_required
from articleapp.forms import ArticleCreationForm
from articleapp.models import Article
from commentapp.forms import CommentCreationForm
import requests
import re
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from pathlib import Path
import os
import random
SEED = 19

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
BASE_DIR = Path(__file__).resolve().parent
device = torch.device("cpu")
path = os.path.join(BASE_DIR, 'model.pth')
print('path',path)


MAX_LEN = 256
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = torch.load(path, map_location=device)
# 기쁨 joy, 사랑 love, 놀람 surprise, 화남 anger, 불안 fear, 슬픔 sadness

def transrate(data):
    url = 'https://translate.kakao.com/translator/translate.json'
    headers = {
        "Referer": "https://translate.kakao.com/",
        "User-Agent": "Mozilla/5.0"
    }
    query = {
        "queryLanguage": "ko",
        "resultLanguage": "en",
        "q": data
    }
    resp = requests.post(url, headers=headers, data=query)
    query = resp.json()
    output = query['result']['output'][0][0]
    return output


@method_decorator(login_required, 'get')
@method_decorator(login_required, 'post')
class ArticleCreateView(CreateView):
    model = Article
    form_class = ArticleCreationForm
    # success_url = reverse_lazy('articleapp:list')
    template_name = 'articleapp/create.html'

    def form_valid(self, form):
        form.instance.writer = self.request.user  # Foreign Key 지정하여 삽입하기 위한 코드
        return super().form_valid(form)

    def get_success_url(self):  # self.object는 target_object와 동일하다고 보면 됨
        obj = Article.objects.get(pk=self.object.id)
        content = obj.content
        content = re.sub('<.>|</.>', '', content)
        transrated_content = transrate(content)

        ####################

        input_ids = torch.tensor(
            [tokenizer.encode(transrated_content, add_special_tokens=True, max_length=MAX_LEN, pad_to_max_length=True)])
        attention_masks = []
        ## Create a mask of 1 for all input tokens and 0 for all padding tokens
        attention_masks = torch.tensor([[float(i > 0) for i in seq] for seq in input_ids])
        logits = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
        y_hat = F.softmax(logits[0][0])
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        result = list(map(float, np.round(y_hat.detach().numpy(), 5) * 100))
        result_dict = {'joy': result[2], 'sadness': result[4], 'fear': result[1], 'surprise': result[5],
                       'anger': result[0], 'love': result[3]}
        print(result_dict)
        print(type(result_dict['sadness']))

        ####################

        # emotion = emotion_analyzer.predict(transrated_content)
        # obj.others = emotion.probas['others']
        # obj.joy = emotion.probas['joy']
        # obj.surprise = emotion.probas['surprise']
        # obj.disgust = emotion.probas['disgust']
        # obj.anger = emotion.probas['anger']
        # obj.sadness = emotion.probas['sadness']
        # obj.fear = emotion.probas['fear']


        obj.joy = result_dict['joy']
        obj.sadness = result_dict['sadness']
        obj.fear = result_dict['fear']
        obj.surprise = result_dict['surprise']
        obj.anger = result_dict['anger']
        obj.love = result_dict['love']

        obj.save()
        return reverse('articleapp:detail', kwargs={'pk': self.object.pk})
#
#
#
# @method_decorator(login_required, 'get')
# @method_decorator(login_required, 'post')
# class ArticleCreateView(CreateView):
#     model = Article
#     form_class = ArticleCreationForm
#     # success_url = reverse_lazy('articleapp:list')
#     template_name = 'articleapp/create.html'
#
#     def form_valid(self, form):
#         form.instance.writer = self.request.user  # Foreign Key 지정하여 삽입하기 위한 코드
#         return super().form_valid(form)
#
#     def get_success_url(self):  # self.object는 target_object와 동일하다고 보면 됨
#         obj = Article.objects.get(pk=self.object.id)
#
#         result=predict(obj.content)
#
#         result_dict = {'joy':result[0], 'sadness':result[1], 'fear':result[2], 'upset':result[3], 'anger':result[4], 'hurt':result[5]}
#
#         obj.joy = result_dict['joy']
#         obj.sadness = result_dict['sadness']
#         obj.fear = result_dict['fear']
#         obj.upset = result_dict['upset']
#         obj.anger = result_dict['anger']
#         obj.hurt = result_dict['hurt']
#
#         obj.save()
#         return reverse('articleapp:detail', kwargs={'pk': self.object.pk})


class ArticleDetailView(DetailView, FormMixin):
    model = Article
    context_object_name = 'target_article'
    template_name = 'articleapp/detail.html'
    form_class = CommentCreationForm


@method_decorator(article_ownership_required, 'get')
@method_decorator(article_ownership_required, 'post')
class ArticleUpdateView(UpdateView):
    model = Article
    form_class = ArticleCreationForm
    context_object_name = 'target_article'
    template_name = 'articleapp/update.html'

    def form_valid(self, form):
        form.instance.writer = self.request.user  # Foreign Key 지정하여 삽입하기 위한 코드
        return super().form_valid(form)

    def get_success_url(self):  # self.object는 target_object와 동일하다고 보면 됨
        obj = Article.objects.get(pk=self.object.id)
        content = obj.content
        content = re.sub('<.>|</.>', '', content)
        transrated_content = transrate(content)

        ####################

        input_ids = torch.tensor(
            [tokenizer.encode(transrated_content, add_special_tokens=True, max_length=MAX_LEN, pad_to_max_length=True)])
        attention_masks = []
        ## Create a mask of 1 for all input tokens and 0 for all padding tokens
        attention_masks = torch.tensor([[float(i > 0) for i in seq] for seq in input_ids])
        logits = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
        y_hat = F.softmax(logits[0][0])
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        result = list(map(float, np.round(y_hat.detach().numpy(), 5) * 100))
        result_dict = {'joy': result[2], 'sadness': result[4], 'fear': result[1], 'surprise': result[5],
                       'anger': result[0], 'love': result[3]}
        print(result_dict)
        print(type(result_dict['sadness']))

        ####################

        # emotion = emotion_analyzer.predict(transrated_content)
        # obj.others = emotion.probas['others']
        # obj.joy = emotion.probas['joy']
        # obj.surprise = emotion.probas['surprise']
        # obj.disgust = emotion.probas['disgust']
        # obj.anger = emotion.probas['anger']
        # obj.sadness = emotion.probas['sadness']
        # obj.fear = emotion.probas['fear']

        obj.joy = result_dict['joy']
        obj.sadness = result_dict['sadness']
        obj.fear = result_dict['fear']
        obj.surprise = result_dict['surprise']
        obj.anger = result_dict['anger']
        obj.love = result_dict['love']

        obj.save()
        return reverse('articleapp:detail', kwargs={'pk': self.object.pk})


class ArticleDeleteView(DeleteView):
    model = Article
    context_object_name = 'target_article'
    success_url = reverse_lazy('articleapp:list')
    template_name = 'articleapp/delete.html'


class ArticleListView(ListView):
    model = Article
    context_object_name = 'article_list'
    template_name = 'articleapp/list.html'
    paginate_by = 20

