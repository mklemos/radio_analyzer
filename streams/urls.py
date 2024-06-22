from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('get_transcriptions/', views.get_transcriptions, name='get_transcriptions'),
]
