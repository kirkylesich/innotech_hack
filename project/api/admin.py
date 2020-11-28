from django.contrib import admin
from api.models import FinUser



@admin.register(FinUser)
class FinUserAdmin(admin.ModelAdmin):
    pass
# Register your models here.
