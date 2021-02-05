from django.urls import path, include
from mlmodel.views import *


urlpatterns = [path('getml/', get_ml)]