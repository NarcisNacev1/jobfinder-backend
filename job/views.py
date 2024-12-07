from rest_framework.response import Response
from rest_framework import status, viewsets
from rest_framework.status import HTTP_201_CREATED
from .serializers import CvSerializer

class CvUpload(viewsets.ViewSet):
    """
    Viewset for handling Cv uploads
    """

    def create(self, request):
        """
        Handling the file upload with user information
        :param request:
        :return:
        """
        # Deserialize the incoming data from the postman request
        serializer = CvSerializer(data=request.data)

        # If the data is valid, save it to serializer and then to the database
        if serializer.is_valid():
            serializer.save()
            # Return the response with status HTTP_201_CREATED
            return Response(serializer.data, status=HTTP_201_CREATED)

        # Return errors if validation fails
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)