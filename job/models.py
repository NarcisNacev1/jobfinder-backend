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
    """
    Basic job information
    """
    title = models.CharField(max_length=100)
    company_name = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    description = models.TextField()
    #Requirements and matching criteria
    skills_required = models.TextField()
    experience_required = models.CharField(max_length=50, blank=True, null=True)
    education_required = models.CharField(max_length=100, blank=True, null=True)
    #Other details
    job_type = models.CharField(
        max_length=50,
        choices=[('Full-Time', 'Full-Time'),
                 ('Part-Time', 'Part-Time'),
                 ('Internship', 'Internship')],
        default='Full-Time'
    )
    salary_range = models.CharField(max_length=50, blank=True, null=True)
    #Metadata
    posted_at = models.DateTimeField(auto_now_add=True)
    source = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"{self.title} at {self.company_name}"
