import requests
from typing import Optional
from config import settings


class AuthClient:
    
    def __init__(self):
        self.base_url = settings.API_BASE_URL
        self.token: Optional[str] = None
    
    def get_token(self) -> str:
        if self.token:
            return self.token
        
        auth_url = f"{self.base_url}{settings.AUTH_ENDPOINT}"
        payload = {
            "clientId": settings.CLIENT_ID,
            "secret": settings.SECRET_KEY
        }
        
        try:
            response = requests.post(auth_url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            token = data.get("access_token")
            
            if not token:
                raise Exception(f"No token found in response. Response keys: {list(data.keys())}")
            
            self.token = token
            return token
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Authentication failed: {str(e)}")
    
    def get_headers(self) -> dict:
        token = self.get_token()
        
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }