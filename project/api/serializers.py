from rest_framework import serializers
from api.models import FinUser

class UserIdFinSerializer(serializers.ModelSerializer):
    

    class Meta:
        model = FinUser
        fields = ['vk_id']


class UserPhotoSerializer(serializers.ModelSerializer):

    class Meta:
        models = FinUser
        fields = ['photo_raw']

