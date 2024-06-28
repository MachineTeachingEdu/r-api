from .views import predict

from django.urls import path 

urlpatterns = [path('predict/', predict)]