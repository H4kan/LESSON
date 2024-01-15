import wandb
import pandas as pd
import os

api = wandb.Api()
entity, project_name = "lesson-research", "LESSON"



runs = api.runs(f"{entity}/{project_name}")

df_dict = {}



for run in runs:
     if run.name.startswith("e "):
       run_name = run.name.split()
       os.makedirs(f"data/{run_name[1]}", exist_ok=True)
       df = run.history(samples=5)
       cols = df.columns
       df_dict[run_name[1]] = {}
       for col in cols:
        if col != 'frames':
            df_dict[run_name[1]][col] = pd.DataFrame()

os.makedirs("data", exist_ok=True)
     
for run in runs:
    if run.name.startswith("e "):
        run_name = run.name.split()
        df = run.history(samples=1e6)
        for col in df.columns:
           if col != 'frames':
            df_dict[run_name[1]][col][run_name[2]] = df[col]
            df_dict[run_name[1]][col]["frames"] = df["frames"]

for env_name, env_runs in df_dict.items():
   for prop_name, prop_results in env_runs.items():
        prop_results.set_index("frames", inplace=True)
        prop_results.to_csv(f"data/{env_name}/{prop_name.replace('/', '_')}.csv")
