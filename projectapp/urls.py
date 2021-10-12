from django.urls import path

from projectapp.views import ProjectCreateView, ProjectDetailView, ProjectListView, ProjectUpdateView, ProjectDeleteView, ProjectCollectionView

app_name='projectapp'

urlpatterns = [
    path('create/', ProjectCreateView.as_view(), name='create'),
    path('detail/<int:pk>', ProjectDetailView.as_view(), name='detail'),
    path('list/', ProjectListView.as_view(), name='list'),
    path('updata/<int:pk>', ProjectUpdateView.as_view(), name='update'),
    path('delete/<int:pk>', ProjectDeleteView.as_view(), name='delete'),
    path('collection/<int:pk>', ProjectCollectionView.as_view(), name='collection'),
]