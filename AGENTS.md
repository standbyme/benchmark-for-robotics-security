I am Hongyu Cai, here is the task I received from my boss. I need to update this project's code.

Message:

Hongyu will look at our external datasets (robo2vlm, droid etc.) to see how many of them already have sequential frames and how to best bring them into our dataset (is our dataset always the start of a sequential frame already or...?)
Hongyu + Jeffery will go over the design document for how a user can use RoboBench downstream to evaluate, to give feedback and criticism

@Hongyu Cai to reiterate what needs to be done. I implemented shawshank for visual attack: https://arxiv.org/pdf/2511.16347 and tried the attack on RoboVQM, it looks okay. The dataset lives as a parquet with all the other attack artifacts and vanilla ones as parquets on mdctleo/RoboJailBench on huggingface right now.

Future attacks and vanilla will result in a parquet and experiments/evaluation should use the parquets.

Shawshank needs to be applied to all the other datasets.
we have a new expanded RJB-Instructions dataset, covering 50 per category. This may be a bit too much and we can potentially subsample 20 per category, up to you. All attacks need to be generated for this new RJB-Instructions
We need to bring in sequential frames from at least one or two dataset and generate attacks on them and run experiments on them.

OPENAI_API_KEY ANTHROPIC_API_KEY and GOOGLE_AI_STUDIO_KEY are set in the .env file.