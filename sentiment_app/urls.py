from django.urls import path
from django.urls.resolvers import URLPattern
from django.conf import settings
from django.conf.urls.static import static

from .views import (IndexView,
                    search_user,
                    AboutView)


urlpatterns = [
    # path("/", IndexView.as_view(), name="index"),
    path("", search_user, name="index"),
    path("about/", AboutView.as_view(), name="about"),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
