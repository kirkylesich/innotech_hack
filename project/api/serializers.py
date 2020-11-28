from rest_framework import serializers
from api.models import FinUser

class UserIdFinSerializer(serializers.ModelSerializer):
    

    class Meta:
        model = FinUser
        fields = ['vk_id']


class UserPhotoSerializer(serializers.ModelSerializer):

    class Meta:
        model = FinUser
        fields = ['photo_raw']


class UserInfoSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = FinUser
        fields = ['vk_id', 'fb_id']
