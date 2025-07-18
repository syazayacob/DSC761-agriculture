from huggingface_hub import hf_hub_download
import joblib

path = hf_hub_download(repo_id="syazayacob/crop_models", filename="Production_RandomForest.pkl")
bundle = joblib.load(path)
print(bundle.keys())  # should show 'model' and 'scaler'