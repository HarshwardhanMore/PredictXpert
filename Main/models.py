from django.db import models

import numpy as np

# Create your models here.

class Data(models.Model):
    file = models.FileField()
    name = models.CharField(max_length=50)

    def __str__(self):
        return self.name

class Model(models.Model):
    name = models.CharField(max_length=20)
    model = models.CharField(max_length=20)
    features = models.CharField(max_length=100)
    labels = models.CharField(max_length=20)

    def __str__(self):
        return self.name+'_'+self.model
    