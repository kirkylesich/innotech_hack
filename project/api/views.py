from django.shortcuts import render
from rest_framework import generics
from api.serializers import UserPhotoSerializer



class PhotoUserFinView(generics.CreateAPIView):
    serializer_class = UserPhotoSerializer


