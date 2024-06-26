# Generated by Django 5.0.6 on 2024-06-22 04:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("streams", "0002_transcription"),
    ]

    operations = [
        migrations.AddField(
            model_name="transcription",
            name="summary",
            field=models.TextField(default=""),
        ),
        migrations.AlterField(
            model_name="radiostation",
            name="name",
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name="radiostation",
            name="url",
            field=models.URLField(max_length=500),
        ),
    ]
