from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
import requests
from .serializers import CvSerializer
from jobfinder_backend.utils import extract_text_from_pdf, extract_section
from jobfinder_backend.settings import api_key
from .models import JobPosition, Cv
from math import sqrt
from collections import Counter

class CvUpload(viewsets.ViewSet):
    """
    Viewset for handling CV uploads
    """
    def create(self, request):
        """
        Handle the file upload, parse the CV for information, and save it.
        """
        serializer = CvSerializer(data=request.data)

        if serializer.is_valid():
            cv_instance = serializer.save()

            cv_file_path = cv_instance.cv_file.path
            cv_text = extract_text_from_pdf(cv_file_path)

            skills_keywords = ["skills", "technologies", "expertise"]
            education_keywords = ["education", "university", "degree"]
            experience_keywords = ["experience", "worked at", "role", "job"]

            skills = extract_section(cv_text, skills_keywords)
            education = extract_section(cv_text, education_keywords)
            experience = extract_section(cv_text, experience_keywords)

            cv_instance.skills = skills
            cv_instance.education = education
            cv_instance.experience = experience
            cv_instance.save()

            return Response({
                "full_name": cv_instance.full_name,
                "email": cv_instance.email,
                "phone_number": cv_instance.phone_number,
                "location": cv_instance.location,
                "skills": skills,
                "education": education,
                "experience": experience,
                "cv_file": cv_instance.cv_file.url
            }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=["post"], url_path="fetch-job")
    def fetch_job(self, request):
        """
        Fetch job data using the scrapingDog API and then fetch job details
        :param request:
        :return:
        """
        field = request.data.get("field", "programming")
        location_id = request.data.get("geoid", "103420483")
        page = request.data.get("page", 1)

        url = "https://api.scrapingdog.com/linkedinjobs"

        params = {
            "api_key": api_key,
            "field": field,
            "geoid": location_id,
            "page": page
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return Response({
                "error": "Request to ScrapingDog API failed",
                "details": str(e),
                "response_text": response.text
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            return Response({
                "error": "Failed to parse JSON from the response",
                "response_text": response.text
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if isinstance(data, list):
            jobs = []
            job_ids = []

            for job in data:
                job_id = job.get("job_id")
                if job_id:
                    job_ids.append(job_id)

                    job_data = {
                        "job_position": job.get("job_position"),
                        "job_link": job.get("job_link"),
                        "company_name": job.get("company_name"),
                        "job_location": job.get("job_location"),
                        "job_posting_date": job.get("job_posting_date"),
                        "job_description": "No description available"
                    }

                    job_instance = JobPosition.objects.create(**job_data)
                    jobs.append(job_instance)

            detailed_jobs = []
            for job_id in job_ids:
                job_details_url = f"https://api.scrapingdog.com/linkedinjobs?api_key={api_key}&job_id={job_id}"
                job_details_response = requests.get(job_details_url)

                if job_details_response.status_code == 200:
                    try:
                        job_details = job_details_response.json()

                        if isinstance(job_details, list):
                            job_details = job_details[0]

                        job_data = {
                            "job_position": job_details.get("job_position", "Not Available"),
                            "job_link": job_details.get("job_link", "Not Available"),
                            "company_name": job_details.get("company_name", "Not Available"),
                            "job_location": job_details.get("job_location", "Not Available"),
                            "job_posting_date": job_details.get("job_posting_time", "Not Available"),
                            "job_description": job_details.get("job_description", "No description available"),
                            "seniority_level": job_details.get("Seniority_level", "Not Available"),
                            "employment_type": job_details.get("Employment_type", "Not Available"),
                            "job_function": job_details.get("Job_function", "Not Available"),
                            "industries": job_details.get("Industries", "Not Available"),
                            "job_apply_link": job_details.get("job_apply_link", "Not Available"),
                        }

                        job_instance = JobPosition.objects.create(**job_data)
                        detailed_jobs.append(job_instance)

                    except requests.exceptions.JSONDecodeError:
                        detailed_jobs.append({
                            "error": f"Failed to parse job details as JSON for job_id {job_id}"
                        })
                else:
                    detailed_jobs.append({
                        "error": f"Failed to fetch job details for job_id {job_id}",
                        "status": job_details_response.status_code
                    })

            return Response({"jobs": "Jobs have been added to the database."}, status=status.HTTP_200_OK)

        else:
            return Response({
                "error": "Unexpected response format from ScrapingDog API. Expected a list.",
                "response_text": response.text
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=["get"], url_path="rank-jobs")
    def rank_jobs(self, request):
        """
        Rank jobs based on relevance to the most recently uploaded CV,
        ensuring unique positions for the same company and only considering jobs with meaningful descriptions.
        """
        # Fetch the most recent CV by uploaded_at
        user_cv = Cv.objects.order_by('-uploaded_at').first()

        if not user_cv:
            return Response({"error": "No CV found."}, status=404)

        # Combine CV data
        cv_data = f"{user_cv.skills or ''} {user_cv.experience or ''} {user_cv.education or ''}"

        # Tokenize and calculate TF-IDF for CV
        cv_tokens = Counter(cv_data.lower().split())
        cv_tfidf = {word: freq / sum(cv_tokens.values()) for word, freq in cv_tokens.items()}

        # Fetch jobs with meaningful descriptions (non-empty and not "No description available")
        jobs = JobPosition.objects.exclude(job_description__isnull=True) \
            .exclude(job_description__exact="") \
            .exclude(job_description__icontains="No description available")

        if not jobs.exists():
            return Response({"error": "No jobs with meaningful descriptions found in the database."}, status=404)

        # Use a set to track unique job_position and company_name pairs
        unique_jobs = {}
        priority_queue = []

        for job in jobs:
            # Generate a unique key for the combination of job_position and company_name
            unique_key = (job.job_position.lower(), job.company_name.lower())
            if unique_key in unique_jobs:
                continue
            unique_jobs[unique_key] = job

            job_data = f"{job.job_description} {job.job_function or ''} {job.industries or ''}"
            job_tokens = Counter(job_data.lower().split())
            job_tfidf = {word: freq / sum(job_tokens.values()) for word, freq in job_tokens.items()}

            # Calculate relevance score using cosine similarity
            dot_product = sum(cv_tfidf[word] * job_tfidf.get(word, 0) for word in cv_tfidf)
            magnitude_cv = sqrt(sum(val ** 2 for val in cv_tfidf.values()))
            magnitude_job = sqrt(sum(val ** 2 for val in job_tfidf.values()))
            relevance_score = (
                    dot_product / (magnitude_cv * magnitude_job) * 100
            ) if magnitude_cv and magnitude_job else 0

            priority_queue.append((relevance_score, job))

        # Sort jobs by relevance score in descending order
        priority_queue.sort(reverse=True, key=lambda x: x[0])

        ranked_jobs = []
        for score, job in priority_queue:
            ranked_jobs.append({
                "job_position": job.job_position,
                "company_name": job.company_name,
                "job_description": (job.job_description[:300] + '...') if len(
                    job.job_description) > 300 else job.job_description,
                "score": round(score, 2),
            })

        return Response({
            "ranked_jobs": ranked_jobs,
            "description": "Score represents the match percentage between your CV and the job description."
        }, status=200)

