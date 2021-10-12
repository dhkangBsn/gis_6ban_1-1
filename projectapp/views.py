from django.contrib.auth.decorators import login_required
from django.shortcuts import render

# Create your views here.
from django.urls import reverse_lazy, reverse
from django.utils.decorators import method_decorator
from django.views.generic import CreateView, DetailView, ListView, UpdateView
from django.views.generic.edit import FormMixin, DeleteView
from django.views.generic.list import MultipleObjectMixin

from articleapp.models import Article
from projectapp.decorators import project_ownership_required
from projectapp.forms import ProjectCreationForm
from projectapp.models import Project
from subscribeapp.models import Subscription


@method_decorator(login_required, 'get')
@method_decorator(login_required, 'post')
class ProjectCreateView(CreateView):
    model = Project
    form_class = ProjectCreationForm
    # success_url = reverse_lazy('articleapp:list') login_required를 적용하였으므로 아래의와 같이 get_success_url 메소드를 재정의한다.
    template_name = 'projectapp/create.html'
    def form_valid(self, form):
        form.instance.writer = self.request.user #request를 보내는 user로 writer를 할당
        return super().form_valid(form)
    def get_success_url(self):
        return reverse("projectapp:detail" , kwargs={"pk":self.object.pk})


class ProjectDetailView(DetailView, MultipleObjectMixin):
    model = Project
    context_object_name = 'target_project'
    template_name = 'projectapp/detail.html'

    paginate_by = 20

    def get_context_data(self, **kwargs):
        user = self.request.user
        project = self.object

        if user.is_authenticated:
            subscription = Subscription.objects.filter(user=user,
                                                       project=project)
        else:
            subscription = None

        article_list = Article.objects.filter(project=self.object)
        lat_lon = article_list.values('lat','lon')
        return super().get_context_data(object_list=article_list,
                                        subscription=subscription, lat_lon=lat_lon ,object_name='article',
                                        **kwargs)

from django.db.models import Avg, Min, Max, Count, F, Sum
class ProjectCollectionView(DetailView):
    model = Project
    context_object_name = 'project_list'
    template_name = 'projectapp/collection.html'
    paginate_by = 20

    def get_context_data(self, **kwargs):
        article_list = Article.objects.filter(project=self.object)
        lat_lon = article_list.values('lat','lon')
        print(lat_lon)
        lat = Article.objects.aggregate(Avg('lat'))
        lon = Article.objects.aggregate(Avg('lon'))
        return super().get_context_data(article_list=article_list,lat_lon=lat_lon,lat=lat,lon=lon,**kwargs)


class ProjectListView(ListView):
    model = Project
    context_object_name = 'project_list'
    template_name = 'projectapp/list.html'
    paginate_by = 20

@method_decorator(project_ownership_required,'get')
@method_decorator(project_ownership_required,'post')
class ProjectUpdateView(UpdateView):
    model = Project
    form_class = ProjectCreationForm
    context_object_name = 'target_project'
    template_name = 'projectapp/update.html'
    def get_success_url(self):
        return reverse('projectapp/detail_article.html', kwargs={'pk':self.object.pk})

@method_decorator(project_ownership_required,'get')
@method_decorator(project_ownership_required,'post')
class ProjectDeleteView(DeleteView):
    model = Project
    context_object_name = 'target_project'
    success_url = reverse_lazy('projectapp:list')
    template_name = 'projectapp/delete.html'