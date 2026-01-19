from django.urls import path
from .views import predict_view, landing_view


urlpatterns = [
    path('', landing_view, name='landing'),
    path('predict/', predict_view, name='predict'),
]
