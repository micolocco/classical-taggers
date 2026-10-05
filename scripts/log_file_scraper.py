import re
from collections import defaultdict


#Helpfull little tool to scrape the sometimes very large snakemake log file and find out which rules jobs are done and which are in progress.

# Path to your log file
log_path = "/home/togasa/classical-taggers/.snakemake.log"

# Read the entire file at once
with open(log_path, "r", encoding="utf-8") as f:
    content = f.read()

# Dictionary: rule_name -> {"job_ids": set(), "done": set()}
rules = defaultdict(lambda: {"job_ids": set(), "done": set()})

# Match blocks: rule name and jobid
rule_pattern = re.compile(
    r"rule\s+(\S+):.*?jobid:\s*(\d+)",
    re.DOTALL
)

# Find all rule-jobid pairs
for match in rule_pattern.finditer(content):
    rule_name, jobid = match.groups()
    rules[rule_name]["job_ids"].add(int(jobid))

# Find finished jobs
finished_jobs = [int(j) for j in re.findall(r"Finished jobid:\s*(\d+)", content)]

#Error in rule
# Find failed jobs
failed_jobs = [int(j) for j in re.findall(r"Error in rule.*?jobid:\s*(\d+)", content, re.DOTALL)]

# Combine finished and failed jobs to mark them as not in progress
not_running_jobs = set(finished_jobs) | set(failed_jobs)

# Mark jobs as done/in progress
for rule_name, data in rules.items():
    data["not running"] = data["job_ids"] & set(not_running_jobs)

# Print results
for rule_name, data in rules.items():
    total = len(data["job_ids"])
    not_running = len(data["not running"])
    done = len(data["not running"] & set(finished_jobs))
    errored = len(data["not running"] & set(failed_jobs))
    in_progress = total - not_running
    print(f"{rule_name}: total={total}, not running={not_running} of that, done={done}, errored={errored}, in_progress={in_progress}")

#Running jobs

for rule_name, data in rules.items():
    print(f"\n{rule_name}:")
    for id in data["job_ids"]:
        if id not in data["not running"]:
            print(f"  Job {id} is not finished.")
