from django.urls import path

from emotionapp.views import EmotionCreateView, EmotionListView, EmotionUpdateView, EmotionDeleteView, EmotionDetailView

app_name='emotionapp'

urlpatterns = [
    path('create/', EmotionCreateView.as_view(), name='create'),
    path('list/', EmotionListView.as_view(), name='list'),
    path('detail/<int:pk>', EmotionDetailView.as_view(), name='detail'),
    path('updata/<int:pk>', EmotionUpdateView.as_view(), name='update'),
    path('delete/<int:pk>', EmotionDeleteView.as_view(), name='delete'),
]