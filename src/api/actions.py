import requests
from typing import List
from config import settings
from src.models import Action
from src.auth.client import AuthClient


class ActionsAPI:
    
    def __init__(self, auth_client: AuthClient):
        self.auth_client = auth_client
        self.base_url = settings.API_BASE_URL
    
    def fetch_actions(self) -> List[Action]:
        actions_url = f"{self.base_url}{settings.ACTIONS_ENDPOINT}"
        
        try:
            response = requests.get(actions_url, headers=self.auth_client.get_headers())
            response.raise_for_status()
            data = response.json()
            
            capabilities = data.get("capabilities", [])
            
            if not capabilities:
                print("Warning: No capabilities found in API response")
                return []
            
            actions = []
            for cap in capabilities:
                action = Action(
                    id=cap.get("id", ""),
                    title=cap.get("title", ""),
                    description=cap.get("description", ""),
                    endpoint=cap.get("endpoint", ""),
                    method=cap.get("method", ""),
                    required_inputs=cap.get("required_inputs", [])
                )
                actions.append(action)
            
            return actions
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch actions: {str(e)}")
        except (KeyError, ValueError) as e:
            raise Exception(f"Failed to parse actions response: {str(e)}")