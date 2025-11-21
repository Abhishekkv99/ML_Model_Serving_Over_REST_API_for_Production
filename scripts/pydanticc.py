from pydantic import BaseModel, HttpUrl

class MyModel(BaseModel):
    url: HttpUrl

# Example of a valid URL
try:
    model = MyModel(url="https://www.shutterstock.com/shutterstock/photos/2665199731/display_1500/stock-photo-full-length-of-trendy-fashion-asian-attractive-pretty-young-woman-model-poses-confidence-wear-zebra-2665199731.jpg")
    print("URL is valid:", model.url)
except ValueError as e:
    print(f"Validation error: {e}")

# Example with userinfo
try:
    model_with_userinfo = MyModel(url='http://user:pass@sub.example.com:8080/resource')
    print("URL with userinfo is valid:", model_with_userinfo.url)
except ValueError as e:
    print(f"Validation error: {e}")