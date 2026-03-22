from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict.html', views.predict_html, name='predict_html'),
    path('predict', views.predict, name='predict'),
    path('health', views.health, name='health'),
]
