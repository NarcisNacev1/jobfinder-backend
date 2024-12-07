from rest_framework.routers import SimpleRouter
from .views import CvUpload

# Create the router instance
job_router = SimpleRouter()

# Register your viewsets
job_router.register(r'cv', CvUpload, basename='cv-upload')

# Include the registered viewsets in the urls
urlpatterns = job_router.urls
