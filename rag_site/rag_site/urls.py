from django.urls import path

from qa.views import ask

urlpatterns = [path("ask/", ask)]
