from django.urls import path, include
from rest_framework import routers
from .views import AudioFileView,TextView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('speech2text/<str:filename>', AudioFileView.as_view(), name='speech2text'),
    path('summary', TextView.as_view(), name='summary'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)