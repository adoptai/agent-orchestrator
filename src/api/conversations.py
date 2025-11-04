import requests
from typing import Optional
from config import settings
from src.auth.client import AuthClient


class ConversationsAPI:
    
    def __init__(self, auth_client: AuthClient):
        self.auth_client = auth_client
        self.base_url = settings.API_BASE_URL
    
    def get_last_user_message(self, conversation_id: str) -> str:
        conversations_url = f"{self.base_url}{settings.CONVERSATIONS_ENDPOINT}".replace(
            "{conversation_id}", 
            conversation_id
        )
        
        try:
            response = requests.get(
                conversations_url, 
                headers=self.auth_client.get_headers()
            )
            
            response.raise_for_status()
            data = response.json()
            messages = data.get("messages", [])
            
            if not messages:
                raise Exception("No messages found in conversation")
            
            for message in reversed(messages):
                role = message.get("role", message.get("sender", ""))
                
                if role.lower() == "user":
                    content = message.get("content")
                    if content:
                        return content
            
            raise Exception("No user message found in conversation")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch conversation: {str(e)}")
        except (KeyError, ValueError) as e:
            raise Exception(f"Failed to parse conversation response: {str(e)}")