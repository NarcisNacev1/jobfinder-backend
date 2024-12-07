from rest_framework import serializers
from .models import Cv

class CvSerializer(serializers.ModelSerializer):
    class Meta:
        model = Cv
        fields = ['full_name', 'email', 'phone_number', 'location', 'cv_file']