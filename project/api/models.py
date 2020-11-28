from django.db import models

# Create your models here.


class FinUser(models.Model):
    vk_id = models.CharField(max_length=250, blank=True)
    fb_id = models.CharField(max_length=250, blank=True)
    photo_embed = models.TextField()
    photo_raw = models.ImageField(upload_to='photos', max_length=254)


