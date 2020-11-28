from django.db import models

# Create your models here.


class FinUser(models.Model):
    vk_id = models.CharField(max_length=250)
    photo_raw = models.BinaryField()
    photo_embed = models.TextField()


