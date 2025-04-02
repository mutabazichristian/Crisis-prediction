from locust import HttpsUser, task, between

 class ModelUser(HttpsUser):
     wait_time = between(1,5)

     @task
     def predict_banking_crisis(self):
         payload = {
                 "country":1,
                 "year":2000,
                 "exch_usd":0.5,
                 "gdp_weighted_default": -0.2,
                 "inflation_annual_cpi": 0.8,
                 "systemic_crisis": 0
                 }
         self.client.post("/predict/api", json=payload)

    @task(weight=3)
    def bulk_predict(self):
        payload = {
                "data":[
                    {
                        "country":1,
                        "year":2000,
                        "exch_usd":0.6,
                        "gdp_weighted_default":-0.2,
                        "inflation_annual_cpi":0.9,
                        "systemic_crisis":0
                        },
                    #TODO: add more entries
                    ]
                }
        self.client.post("predict/batch",json=payload)
