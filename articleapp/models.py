from django.contrib.auth.models import User
from django.db import models

# Create your models here.
from projectapp.models import Project
from imagekit.models import ProcessedImageField
from imagekit.processors import ResizeToFill

class Article(models.Model):
    writer = models.ForeignKey(User, on_delete=models.SET_NULL, related_name='article', null=True)
    # User가 탈퇴했을 경우, 작성자 미상(게시글에서) 처럼 되게 한다.
    # OneToOne과 다른 점은 다대다도 가능하다.
    title = models.CharField(max_length=200)
    image = models.ImageField(upload_to='article/',null=True)
    # image = ProcessedImageField(
    #     upload_to='article/',  # 저장 위치
    #     processors=[ResizeToFill(600, 600)],  # 처리할 작업 목록
    #     format='JPEG',  # 저장 포맷(확장자)
    #     options={'quality': 90},  # 저장 포맷 관련 옵션 (JPEG 압축률 설정)
    # )
    content = models.TextField(null=True)
    created_at = models.DateField(auto_now_add=True, null=True)

    # 어떤 게시글이 연길되어 있는지 설정
    # 게시글과 프로젝트의 연결고리
    project = models.ForeignKey(Project, on_delete=models.SET_NULL, related_name='article', null=True, blank=True)

    # likeapp 과 연결된 칼럼
    like = models.IntegerField(default=0)  # 새로운 글을 썼을 때, default 0 으로 자동 저장됨.

    # emotion : sadness, anger, love, surprise, fear, joy
    # 기쁨 0 joy
    #
    # 슬픔 1 sadness
    #
    # 불안 2 fear
    #
    # 당황 3 upset
    #
    # 분노 4 anger
    #
    # 상처 5 hurt
    # joy, sadness, fear, upset, anger, hurt
    # 기쁨, 슬픔, 불안, 당항, 분노, 상처
    joy = models.FloatField(null=True)
    sadness = models.FloatField(null=True)
    fear = models.FloatField(null=True)
    upset = models.FloatField(null=True)
    anger = models.FloatField(null=True)
    hurt = models.FloatField(null=True)

    # 위도, 경도
    lat = models.FloatField(null=False)
    lon = models.FloatField(null=False)

    # 장소
    place = models.CharField(max_length=100, null=True)

    # 공개, 비공개
    is_private = models.BooleanField(default=False, null=False)

