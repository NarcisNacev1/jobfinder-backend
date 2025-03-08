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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import numpy as np
from functools import lru_cache

load_dotenv()

# Lazy loading of BERT model
_bert_tokenizer = None
_bert_model = None


def get_bert_model():
    """Lazy load BERT model and tokenizer"""
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        _bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if _bert_model is None:
        _bert_model = BertModel.from_pretrained("bert-base-uncased")
    return _bert_tokenizer, _bert_model


@lru_cache(maxsize=128)
def get_bert_embedding(text):
    """
    Generate BERT embeddings for a given text
    Uses caching to avoid recomputing embeddings for the same text
    """
    tokenizer, model = get_bert_model()
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

            # Extract all sections at once
            cv_instance.skills = extract_section(cv_text, skills_keywords)
            cv_instance.education = extract_section(cv_text, education_keywords)
            cv_instance.experience = extract_section(cv_text, experience_keywords)
            cv_instance.save()

            return Response({
                "full_name": cv_instance.full_name,
                "email": cv_instance.email,
                "phone_number": cv_instance.phone_number,
                "location": cv_instance.location,
                "skills": cv_instance.skills,
                "education": cv_instance.education,
                "experience": cv_instance.experience,
                "cv_file": cv_instance.cv_file.url
            }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=["post"], url_path="fetch-job")
    def fetch_job(self, request):
        """
        Fetch job data using the scrapingDog API and then fetch job details
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
            data = response.json()
        except requests.exceptions.RequestException as e:
            return Response({
                "error": "Request to ScrapingDog API failed",
                "details": str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
        except requests.exceptions.JSONDecodeError:
            return Response({
                "error": "Failed to parse JSON from the response",
                "response_text": response.text if 'response' in locals() else "No response"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if not isinstance(data, list):
            return Response({
                "error": "Unexpected response format from ScrapingDog API. Expected a list.",
                "response_text": str(data)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Track processed job IDs to avoid duplicates
        processed_job_ids = set()
        job_ids_to_fetch = []

        # First pass: Store basic job info and collect job IDs for detailed fetching
        for job in data:
            job_id = job.get("job_id")
            if not job_id or job_id in processed_job_ids:
                continue

            processed_job_ids.add(job_id)
            job_ids_to_fetch.append(job_id)

            # Only store basic info, don't create full job instances yet
            JobPosition.objects.update_or_create(
                job_id=job_id,
                defaults={
                    "job_position": job.get("job_position", "Not Available"),
                    "job_link": job.get("job_link", "Not Available"),
                    "company_name": job.get("company_name", "Not Available"),
                    "job_location": job.get("job_location", "Not Available"),
                    "job_posting_date": job.get("job_posting_date", "Not Available"),
                    "job_description": "Fetching description..."
                }
            )

        # Second pass: Fetch detailed info for all jobs
        detailed_jobs = []

        # Create batches for more efficient API calls (optional)
        batch_size = 5
        for i in range(0, len(job_ids_to_fetch), batch_size):
            batch_ids = job_ids_to_fetch[i:i + batch_size]

            for job_id in batch_ids:
                job_details_url = "https://api.scrapingdog.com/linkedinjobs"
                job_details_params = {
                    "api_key": api_key,
                    "job_id": job_id
                }

                try:
                    job_details_response = requests.get(job_details_url, params=job_details_params)
                    job_details_response.raise_for_status()
                    job_details = job_details_response.json()

                    if isinstance(job_details, list) and job_details:
                        job_details = job_details[0]

                    # Update existing job record with detailed information
                    job_instance = JobPosition.objects.filter(job_id=job_id).first()
                    if job_instance:
                        job_instance.job_position = job_details.get("job_position", job_instance.job_position)
                        job_instance.job_link = job_details.get("job_link", job_instance.job_link)
                        job_instance.company_name = job_details.get("company_name", job_instance.company_name)
                        job_instance.job_location = job_details.get("job_location", job_instance.job_location)
                        job_instance.job_posting_date = job_details.get("job_posting_time",
                                                                        job_instance.job_posting_date)
                        job_instance.job_description = job_details.get("job_description", "No description available")
                        job_instance.seniority_level = job_details.get("Seniority_level", "Not Available")
                        job_instance.employment_type = job_details.get("Employment_type", "Not Available")
                        job_instance.job_function = job_details.get("Job_function", "Not Available")
                        job_instance.industries = job_details.get("Industries", "Not Available")
                        job_instance.job_apply_link = job_details.get("job_apply_link", "Not Available")
                        job_instance.save()

                        detailed_jobs.append(job_instance)

                except (requests.exceptions.RequestException, requests.exceptions.JSONDecodeError) as e:
                    # Log the error but continue processing other jobs
                    print(f"Error fetching details for job {job_id}: {str(e)}")
                    continue

        return Response(
            {"jobs": f"Successfully processed {len(detailed_jobs)} jobs."},
            status=status.HTTP_200_OK
        )

    @action(detail=False, methods=["get"], url_path="rank-jobs")
    def rank_jobs(self, request):
        """
        Rank jobs using two distinct methods:
        Method A: Pure cosine similarity on TF-IDF vectors
        Method B: BERT embeddings with semantic similarity
        """
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not os.getenv("OPENAI_API_KEY"):
                return Response({"error": "OpenAI API key not configured"}, status=500)

            user_cv = Cv.objects.order_by('-uploaded_at').first()
            if not user_cv:
                return Response({"error": "No CV found."}, status=404)

            # Get CV text for both methods
            cv_text = f"{user_cv.skills} {user_cv.experience} {user_cv.education}"

            # Get jobs with meaningful descriptions
            jobs = JobPosition.objects.exclude(job_description__isnull=True) \
                .exclude(job_description__exact="") \
                .exclude(job_description__icontains="No description available") \
                .exclude(job_description__icontains="Fetching description...")

            if not jobs.exists():
                return Response({"error": "No jobs with meaningful descriptions found."}, status=404)

            # Deduplicate jobs based on position, company, and description
            unique_jobs = {}
            for job in jobs:
                key = (job.job_position, job.company_name, job.job_description)
                if key not in unique_jobs:
                    unique_jobs[key] = job
            jobs = list(unique_jobs.values())

            # Create text representations for all jobs
            job_texts = []
            job_objects = []

            for job in jobs:
                job_text = f"{job.job_description} {job.job_function or ''} {job.industries or ''}"
                job_texts.append(job_text)
                job_objects.append(job)

            # METHOD A: TF-IDF + Cosine Similarity (traditional NLP approach)
            vectorizer = TfidfVectorizer(stop_words='english')

            # Create document-term matrix
            all_texts = [cv_text] + job_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Get CV vector (first row) and job vectors (remaining rows)
            cv_vector = tfidf_matrix[0:1]
            job_vectors = tfidf_matrix[1:]

            # Calculate cosine similarity between CV and all jobs
            cosine_similarities = sklearn_cosine_similarity(cv_vector, job_vectors).flatten()

            # Create sorted list of (similarity, job) pairs
            cosine_scores = [(score, job_objects[i]) for i, score in enumerate(cosine_similarities)]
            cosine_scores.sort(key=lambda x: x[0], reverse=True)

            # Prevent duplicate recommendations
            top_cosine_job_ids = set()
            unique_cosine_scores = []

            for score, job in cosine_scores:
                if job.id not in top_cosine_job_ids:
                    top_cosine_job_ids.add(job.id)
                    unique_cosine_scores.append((score, job))
                    if len(unique_cosine_scores) >= 3:
                        break

            top_cosine_jobs = unique_cosine_scores

            # Check if we have enough unique jobs
            if len(top_cosine_jobs) < 3:
                return Response({"error": "Not enough unique jobs to rank."}, status=404)

            # METHOD B: BERT Embeddings with Semantic Similarity
            # Get BERT embedding for CV
            cv_embedding = get_bert_embedding(cv_text)

            bert_scores = []
            for i, job_text in enumerate(job_texts):
                # Get BERT embedding for job
                job_embedding = get_bert_embedding(job_text)

                # Calculate similarity using BERT embeddings
                similarity = torch.nn.functional.cosine_similarity(cv_embedding, job_embedding)
                bert_scores.append((similarity.item(), job_objects[i]))

            # Sort by the similarity score
            bert_scores.sort(key=lambda x: x[0], reverse=True)

            # Prevent duplicate recommendations
            top_bert_job_ids = set()
            unique_bert_scores = []

            for score, job in bert_scores:
                if job.id not in top_bert_job_ids:
                    top_bert_job_ids.add(job.id)
                    unique_bert_scores.append((score, job))
                    if len(unique_bert_scores) >= 3:
                        break

            top_bert_jobs = unique_bert_scores

            # Function to safely truncate text
            def safe_truncate(text, length=500):
                if text and isinstance(text, str):
                    return text[:length] + "..." if len(text) > length else text
                return "Not available"

            # Normalize scores for better comparison
            def normalize_scores(scores, min_val=0, max_val=100):
                if not scores:
                    return []
                min_score = min(scores)
                max_score = max(scores)
                if max_score == min_score:  # Avoid division by zero
                    return [max_val] * len(scores)
                return [min_val + (score - min_score) * (max_val - min_val) / (max_score - min_score) for score in
                        scores]

            # Extract just the scores for normalization
            cosine_scores_values = [score for score, _ in top_cosine_jobs]
            bert_scores_values = [score for score, _ in top_bert_jobs]

            # Normalize scores
            normalized_cosine = normalize_scores(cosine_scores_values)
            normalized_bert = normalize_scores(bert_scores_values)

            # Prepare the prompt for the AI judge with normalized scores
            openai_prompt = f"""
            You are an impartial AI judge for NextMatch, a job recommender system. Your task is to evaluate how well a candidate's 
            resume matches job vacancies. The system has already calculated similarity scores (0-100) using two different methods.

            Input:
            Candidate's CV: 
            {cv_text}

            Method A - Top 3 Job Matches (TF-IDF with Cosine Similarity):
            1. {top_cosine_jobs[0][1].job_position} at {top_cosine_jobs[0][1].company_name} (Score: {normalized_cosine[0]:.2f})
               Description: {safe_truncate(top_cosine_jobs[0][1].job_description)}

            2. {top_cosine_jobs[1][1].job_position} at {top_cosine_jobs[1][1].company_name} (Score: {normalized_cosine[1]:.2f})
               Description: {safe_truncate(top_cosine_jobs[1][1].job_description)}

            3. {top_cosine_jobs[2][1].job_position} at {top_cosine_jobs[2][1].company_name} (Score: {normalized_cosine[2]:.2f})
               Description: {safe_truncate(top_cosine_jobs[2][1].job_description)}

            Method B - Top 3 Job Matches (BERT Semantic Embeddings):
            1. {top_bert_jobs[0][1].job_position} at {top_bert_jobs[0][1].company_name} (Score: {normalized_bert[0]:.2f})
               Description: {safe_truncate(top_bert_jobs[0][1].job_description)}

            2. {top_bert_jobs[1][1].job_position} at {top_bert_jobs[1][1].company_name} (Score: {normalized_bert[1]:.2f})
               Description: {safe_truncate(top_bert_jobs[1][1].job_description)}

            3. {top_bert_jobs[2][1].job_position} at {top_bert_jobs[2][1].company_name} (Score: {normalized_bert[2]:.2f})
               Description: {safe_truncate(top_bert_jobs[2][1].job_description)}

            Please provide an analysis in **valid JSON format** with the following structure:
            {{
              "methodAssessment": {{
                "betterMethod": "A or B",
                "justification": "Explanation of which method provided better matches"
              }},
              "adjustedMatchScores": {{
                "1": {{
                  "jobTitle": "Job Title",
                  "adjustedScore": 0-100
                }},
                "2": {{
                  "jobTitle": "Job Title",
                  "adjustedScore": 0-100
                }},
                "3": {{
                  "jobTitle": "Job Title",
                  "adjustedScore": 0-100
                }}
              }},
              "bestMatch": {{
                "jobTitle": "Job Title",
                "justification": {{
                  "skillsAlignment": "Explanation of skills alignment",
                  "experienceRelevance": "Explanation of experience relevance",
                  "careerGrowthPotential": "Explanation of career growth potential"
                }}
              }}
            }}
            """

            response = client.chat.completions.create(
                model="gpt-4",  # Use a model that supports JSON output
                messages=[{"role": "user", "content": openai_prompt}],
            )

            # Debug the OpenAI response
            import json
            print("OpenAI Response Content:", response.choices[0].message.content)

            # Parse the AI recommendation
            try:
                ai_recommendation = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                ai_recommendation = {
                    "error": "The AI recommendation could not be parsed as JSON.",
                    "raw_response": response.choices[0].message.content
                }

            # Format the results for the frontend
            method_a_jobs = [
                {
                    "job_position": job[1].job_position,
                    "company_name": job[1].company_name,
                    "job_description": safe_truncate(job[1].job_description, 300),
                    "raw_score": job[0],
                    "score": round(normalized_cosine[i], 2),  # Use normalized scores
                    "apply_link": job[1].job_apply_link,
                    "job_id": job[1].id
                }
                for i, job in enumerate(top_cosine_jobs)
            ]

            method_b_jobs = [
                {
                    "job_position": job[1].job_position,
                    "company_name": job[1].company_name,
                    "job_description": safe_truncate(job[1].job_description, 300),
                    "raw_score": job[0],
                    "score": round(normalized_bert[i], 2),  # Use normalized scores
                    "apply_link": job[1].job_apply_link,
                    "job_id": job[1].id
                }
                for i, job in enumerate(top_bert_jobs)
            ]

            return Response({
                "method_a_jobs": method_a_jobs,
                "method_b_jobs": method_b_jobs,
                "ai_recommendation": ai_recommendation,  # Return parsed JSON or error
            }, status=200)

        except Exception as e:
            # Add comprehensive error handling
            import traceback
            return Response({
                "error": f"An error occurred during job ranking: {str(e)}",
                "traceback": traceback.format_exc()
            }, status=500)