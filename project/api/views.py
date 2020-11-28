from django.shortcuts import render
from rest_framework import generics
from api.serializers import UserInfoSerializer, UserPhotoSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from api.models import FinUser




class PhotoUserFinView(APIView):

    def post(self, request):
        serializer = UserPhotoSerializer(data = request.data)
        users = FinUser.objects.all()
        users_serializer = UserInfoSerializer(data = users, many=True)
        if users_serializer.is_valid():
            pass
        if serializer.is_valid():
            return Response(users_serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


