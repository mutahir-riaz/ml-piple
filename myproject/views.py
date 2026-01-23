from django.http import JsonResponse

def hello_api(request):
    return JsonResponse({"message": "Hello! This is my first GET API 🎉"})
