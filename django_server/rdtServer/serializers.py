from rest_framework import serializers
from .models import Align


class AlignSerializer(serializers.ModelSerializer):
    class Meta:
        model = Align
        fields = ("name")