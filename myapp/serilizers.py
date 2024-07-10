from rest_framework import serializers
from .models import emp_data

class seril(serializers.ModelSerializer):
    class Meta:
        model = emp_data
        fields = "__all__"
