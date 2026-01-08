# XG Platform
 
## What it does
- Trains an xG probability model (logistic regression pipeline)
- Serves real-time predictions via FastAPI
- Logs inference requests for monitoring/drift analysis

## Why it exists

I decided to combine my knowledge and understanding of ML with my love for soccer and find an intersection between the two. I trained an xG model using StatsBomb data, I ran through several model choices and settled on Logistic Regression. For me, the natural next step was coming up with a platform to serve the model and monitor it.

## What's next

I plan to add more features to the platform, such as a web interface to view the model's predictions and monitor its performance. I also plan to add more features to the model like defensive pressure, and maybe break it down more by player, team, and opponent.

## How to run

After cloning the repository, you can run the following commands:

```bash
pip install -r requirements.txt
uvicorn src.inference.app:app --reload
```

You should then be able to make requests to the API at `http://localhost:8000` using a new terminal session. Follow this template:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"x":102.4,"y":41.2,"shot_body_part":"Left Foot","shot_type":"Open Play","play_pattern":"Regular Play"}'
```
