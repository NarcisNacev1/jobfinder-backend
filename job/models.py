from django.db import models

# Create your models here.
class Cv(models.Model):
    """
    Cv model to store the cv details and related information
    """
    full_name = models.CharField(max_length=100)
    email = models.EmailField()
    phone_number = models.CharField(max_length=13, blank=True, null=True)
    location = models.CharField(max_length=100, blank=True, null=True)
    #Professional skills
    skills = models.TextField()
    education = models.TextField(blank=True, null=True)
    experience = models.TextField(blank=True, null=True)
    #Metadata
    cv_file = models.FileField(upload_to='cvs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.full_name


class JobPosition(models.Model):
    external_job_id = models.CharField(max_length=100, null=True, blank=True, unique=True)
    job_position = models.CharField(max_length=255)
    company_name = models.CharField(max_length=255)
    job_location = models.CharField(max_length=255)
    job_posting_date = models.CharField(max_length=255, blank=True, null=True)
    job_description = models.TextField(blank=True, null=True)
    job_link = models.URLField(max_length=500)
    seniority_level = models.CharField(max_length=100, blank=True, null=True)
    employment_type = models.CharField(max_length=100, blank=True, null=True)
    job_function = models.CharField(max_length=100, blank=True, null=True)
    industries = models.CharField(max_length=100, blank=True, null=True)
    job_apply_link = models.URLField(max_length=500, blank=True, null=True)

    def __str__(self):
        return f"{self.job_position} at {self.company_name}"




