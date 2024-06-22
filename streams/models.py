from django.db import models

class RadioStation(models.Model):
    name = models.CharField(max_length=255)
    url = models.URLField(max_length=500)  # Ensure this is correct

class Transcription(models.Model):
    station = models.ForeignKey(RadioStation, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    text = models.TextField()
