# import os
# import json
# import kagglehub





# os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
# with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
#     json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)

# os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)


# try:
#     path = kagglehub.model_download("semyaniel/toxic-comment-class/pyTorch/default")
#     print("Path to model files:", path)
# except Exception as e:
#     print("Ошибка:", e)