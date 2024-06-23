from django.db import models

class RadioStation(models.Model):
    name = models.CharField(max_length=255)
    url = models.URLField(max_length=500)

    def __str__(self):
        return self.name

class Transcription(models.Model):
    station = models.ForeignKey(RadioStation, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    text = models.TextField()
    summary = models.TextField(default='')

    def __str__(self):
        return f"{self.station.name} - {self.timestamp}"

class Segment(models.Model):
    station = models.ForeignKey(RadioStation, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()
    audio_path = models.CharField(max_length=255)
    text = models.TextField()
    summary = models.TextField()

    def __str__(self):
        return f"{self.station.name} - {self.timestamp}"