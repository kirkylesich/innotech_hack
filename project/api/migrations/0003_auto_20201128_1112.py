# Generated by Django 3.1.3 on 2020-11-28 11:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_auto_20201128_1111'),
    ]

    operations = [
        migrations.AlterField(
            model_name='finuser',
            name='photo_embed',
            field=models.TextField(),
        ),
        migrations.AlterField(
            model_name='finuser',
            name='photo_raw',
            field=models.ImageField(max_length=254, upload_to='photos'),
        ),
        migrations.AlterField(
            model_name='finuser',
            name='vk_id',
            field=models.CharField(blank=True, max_length=250),
        ),
    ]