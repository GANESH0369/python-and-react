from django.db import models

# Create your models here.
class emp_data(models.Model):
    fname=models.CharField( max_length=50)
    lname=models.CharField( max_length=50)
    email=models.CharField( max_length=50)
    password=models.CharField( max_length=50)


# models.py
from django.db import models

class DataPoint(models.Model):
    feature1 = models.FloatField()
    feature2 = models.FloatField()
    prediction = models.FloatField(null=True, blank=True)
