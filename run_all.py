import subprocess
import os, time

start_time = time.time()  # 开始计时

python_path = "C:/Users/86186/AppData/Local/Programs/Python/Python312/python.exe"

notebooks = [
    "ScoreSample-SEAL/score_gnn.ipynb",
    "ScoreSample-SEAL/score_sample.ipynb",
    "ScoreSample-SEAL/ss_seal.ipynb",
]

for nb in notebooks:
    subprocess.run(
        [
            python_path,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            nb,
        ]
    )

end_time = time.time()  # 结束计时
elapsed_time = end_time - start_time
print(f"Total elapsed time: {elapsed_time:.2f} 秒")
