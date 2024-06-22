from django.db import models

class RadioStation(models.Model):
    name = models.CharField(max_length=100)
    url = models.URLField()

    def __str__(self):
        return self.name

class Transcription(models.Model):
    station = models.ForeignKey(RadioStation, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    text = models.TextField()

    def __str__(self):
        return f"{self.station.name} - {self.timestamp}"
