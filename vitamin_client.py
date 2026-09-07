import io
import requests

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
        
        # VITAMIN richiede multipart/form-data con un file .txt
        files = {
            "file": ("model.txt", io.BytesIO(model_text.encode("utf-8")), "text/plain")
        }
        data = {
            "logic": logic,
            "formula": formula,
            "generate_trace": "true" if generate_trace else "false"
        }

        try:
            response = requests.post(url, headers=headers, data=data, files=files, timeout=5)
            if response.status_code == 200:
                res_json = response.json()
                # Il campo booleano principale è 'satisfied'
                return res_json.get("satisfied", False), res_json
            else:
                print(f"[VITAMIN API ERROR] Status {response.status_code}: {response.text}")
                return False, {"error": response.text}
        except Exception as e:
            print(f"[VITAMIN CONNECTION ERROR]: {e}")
            return False, {"error": str(e)}

    def verify_batch(self, model_text, formulas, logic="CTL"):
        """Invia più formule separate da punto e virgola a /execute-batch."""
        url = f"{self.base_url}/execute-batch"
        headers = self._get_headers()
        
        # Formattazione richiesta: ogni riga termina con ';'
        batch_text = "\n".join([f"{f};" for f in formulas])
        
        files = {
            "file": ("model.txt", io.BytesIO(model_text.encode("utf-8")), "text/plain")
        }
        data = {
            "logic": logic,
            "formulas": batch_text,
            "generate_trace": "false"
        }

        try:
            response = requests.post(url, headers=headers, data=data, files=files, timeout=6)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"[VITAMIN BATCH ERROR]: {e}")
            return []