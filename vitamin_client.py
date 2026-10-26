import io
import os
import json
import requests
import hashlib

from pyFighters import Action
from CTL_bridge import build_horizon_tree, generate_vitamin_model

class VitaminClient:
    def __init__(self, api_key=None, cache_file="vitamin_cache.json"):
        self.base_url = "https://vitamin.r2.enst.fr/api/v1/model-checking"
        self.api_key = api_key
        self.cache_file = cache_file
        
        self.cache = {}
        self.cache_hits = 0
        self.api_calls = 0
        self.unsaved_changes = 0
        
        self._load_cache()

    def _load_cache(self):
        """Carica il database locale se esiste."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"💽 [VITAMIN CACHE] Caricati {len(self.cache)} DAG/Formule da disco.")
            except Exception as e:
                print(f"⚠️ [VITAMIN CACHE] Errore lettura disco ({e}). Inizializzazione pulita.")
                self.cache = {}

    def save_cache(self):
        """Forza il salvataggio su disco del dizionario corrente."""
        if self.unsaved_changes > 0:
            try:
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f)
                self.unsaved_changes = 0
            except Exception as e:
                print(f"⚠️ [VITAMIN CACHE] Fallito salvataggio su disco: {e}")

    def _get_headers(self):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def verify_formula(self, model_text, formula, logic="CTL", generate_trace=False):
        request_string = f"{model_text}_|||_{formula}_|||_{logic}_|||_{generate_trace}"
        request_hash = hashlib.md5(request_string.encode('utf-8')).hexdigest()
        
        if request_hash in self.cache:
            self.cache_hits += 1
            # Il JSON restituisce liste, le riconvertiamo in tuple in uscita
            cached_data = self.cache[request_hash]
            return cached_data[0], cached_data[1]
            
        self.api_calls += 1
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
            response = requests.post(url, headers=headers, data=data, files=files, timeout=15)
            
            if response.status_code == 200:
                res_json = response.json()
                
                if "error" in res_json and res_json["error"]:
                    print(f"[VITAMIN LOGIC ERROR]: {res_json['error']}")
                    result = (False, res_json)
                else:
                    verification = res_json.get("verification", {})
                    is_satisfied = verification.get("initial_state_satisfied", False)
                    result = (is_satisfied, res_json)
                    
                self.cache[request_hash] = result
                self.unsaved_changes += 1
                
                # Autosave ogni 50 nuove formule scoperte
                if self.unsaved_changes >= 50:
                    self.save_cache()
                    
                return result
                
            else:
                print(f"[VITAMIN API ERROR] Status {response.status_code}: {response.text}")
                return False, {"error": response.text}
                
        except Exception as e:
            print(f"[VITAMIN CONNECTION ERROR]: {e}")
            return False, {"error": str(e)}

    def print_cache_stats(self):
        total = self.cache_hits + self.api_calls
        if total > 0:
            saving = (self.cache_hits / total) * 100
            print(f"🧠 [VITAMIN CACHE] Hit: {self.cache_hits} | API Calls: {self.api_calls} | Risparmio: {saving:.1f}%")