from django.urls import path
from . import views

urlpatterns = [
    path('predict_models', views.predict_models),
]