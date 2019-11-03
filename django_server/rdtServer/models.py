from django.db import models

# Create your models here.

class Align(models.Model):
	#name
	name = "test"
	
	def __str__(self):
		return "{} - {}".format(self.name)
