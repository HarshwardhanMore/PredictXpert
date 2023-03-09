from django.urls import path

from Main import views


from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.intro),
    path('home', views.home2),
    path('plots', views.plots),
    path('ml', views.ml),
    path('seeavailabledatasets', views.seeavailabledatasets),
    path('documentations', views.documentations),
    path('rateus', views.rateus),
    path('predict', views.predict),
    path('manualpredict', views.manualpredict),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
