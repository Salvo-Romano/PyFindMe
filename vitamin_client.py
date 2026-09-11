import io
import requests

from pyFighters import Action
from CTL_bridge import build_horizon_tree, generate_vitamin_model


class VitaminClient:
    def __init__(self, api_key=None):
        self.base_url = "https://vitamin.r2.enst.fr/api/v1/model-checking"
        self.api_key = api_key

    def _get_headers(self):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def verify_formula(self, model_text, formula, logic="CTL", generate_trace=False):
        url = f"{self.base_url}/execute"
        headers = self._get_headers()
        
        files = {
            "file": ("model.txt", io.BytesIO(model_text.encode("utf-8")), "text/plain")
        }
        data = {
            "logic": logic,
            "formula": formula,
            "generate_trace": "true" if generate_trace else "false"
        }

        try:
            # Timeout a 15 secondi per le query formali complesse
            response = requests.post(url, headers=headers, data=data, files=files, timeout=15)
            
            if response.status_code == 200:
                res_json = response.json()
                
                if "error" in res_json and res_json["error"]:
                    print(f"[VITAMIN LOGIC ERROR]: {res_json['error']}")
                    return False, res_json
                
                # Accediamo alla struttura corretta
                verification = res_json.get("verification", {})
                is_satisfied = verification.get("initial_state_satisfied", False)
                
                return is_satisfied, res_json
            else:
                print(f"[VITAMIN API ERROR] Status {response.status_code}: {response.text}")
                return False, {"error": response.text}
                
        except Exception as e:
            print(f"[VITAMIN CONNECTION ERROR]: {e}")
            return False, {"error": str(e)}