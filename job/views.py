from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
import requests
from .serializers import CvSerializer
from jobfinder_backend.utils import extract_text_from_pdf, extract_section
from jobfinder_backend.settings import api_key
from .models import JobPosition, Cv
from transformers import BertTokenizer, BertModel
import torch
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


def get_bert_embedding(text):
    """
    Generate BERT embeddings for a given text
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

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
        Rank jobs using BERT embeddings and get OpenAI recommendation for the top 3.
        """
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        user_cv = Cv.objects.order_by('-uploaded_at').first()
        if not user_cv:
            return Response({"error": "No CV found."}, status=404)

        cv_embedding = get_bert_embedding(f"{user_cv.skills} {user_cv.experience} {user_cv.education}")

        jobs = JobPosition.objects.exclude(job_description__isnull=True) \
            .exclude(job_description__exact="") \
            .exclude(job_description__icontains="No description available")

        if not jobs.exists():
            return Response({"error": "No jobs with meaningful descriptions found."}, status=404)

        job_scores = []
        for job in jobs:
            job_embedding = get_bert_embedding(f"{job.job_description} {job.job_function or ''} {job.industries or ''}")
            similarity = torch.nn.functional.cosine_similarity(cv_embedding, job_embedding)
            job_scores.append((similarity.item(), job))  # Use .item() to convert tensor to scalar

        job_scores.sort(reverse=True, key=lambda x: x[0])
        top_jobs = job_scores[:3]

        openai_prompt = f"""
        The following is the candidate's CV skills, experience, and education:

        {user_cv.skills + user_cv.experience + user_cv.education}

        Here are the top 3 job matches for the candidate:
        1. {top_jobs[0][1].job_position} at {top_jobs[0][1].company_name} (Score: {round(top_jobs[0][0], 2)}%)
           Description: {top_jobs[0][1].job_description[:500]}

        2. {top_jobs[1][1].job_position} at {top_jobs[1][1].company_name} (Score: {round(top_jobs[1][0], 2)}%)
           Description: {top_jobs[1][1].job_description[:500]}

        3. {top_jobs[2][1].job_position} at {top_jobs[2][1].company_name} (Score: {round(top_jobs[2][0], 2)}%)
           Description: {top_jobs[2][1].job_description[:500]}

        Based on the candidate's CV, which job is the best fit? Consider skills, experience, and career growth.
        Rank them on a scale of 1-10, with 10 being the best fit.
        
        Also when returning the data dont wrap it in any ** to try to make it look better in the response just use 1. and -. 
        also use full %
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": openai_prompt}],
        )
        ai_recommendation = response.choices[0].message.content

        ranked_jobs = [
            {
                "job_position": job[1].job_position,
                "company_name": job[1].company_name,
                "job_description": job[1].job_description[:300] + "...",
                "score": round(job[0], 2),
                "apply_link": job[1].job_apply_link
            }
            for job in top_jobs
        ]

        return Response({
            "ranked_jobs": ranked_jobs,
            "ai_recommendation": ai_recommendation,
        }, status=200)