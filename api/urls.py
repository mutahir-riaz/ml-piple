from django.urls import path
from . import views

urlpatterns = [
    path('hello/', views.hello_api, name='hello_api'),
    path('upload-csv/', views.upload_csv, name='upload_csv'),
    path('simple-option/', views.simple_option, name='simple_option'),
    path("preprocess-data/", views.preprocess_data, name="preprocess_data"),
    # path('csv-del/', views.del_csv, name='del_csv'),
]