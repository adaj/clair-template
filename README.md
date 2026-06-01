# Clair 

<img src="./visual-identity/clair-a-celadon-removebg-big.png" alt="Avatar" width="100"/><img src="./visual-identity/clair-b-celadon-removebg-big.png" alt="Avatar" width="100"/><img src="./visual-identity/clair-c-celadon-big-removebg.png" alt="Avatar" width="100"/>

## Overview 📖

Clair is a collaborative conversational agent that uses learning analytics and a fuzzy expert system to trigger productive talk moves to facilitate student-student dialogue based on the [Accountable Talk](https://nsiexchange.org/wp-content/uploads/2019/02/AT-SOURCEBOOK2016-1-23-19.pdf) (or APT) framework.

If you need further support or have any questions, join the Clair community on
[Discord](https://discord.gg/kS3t75pkFZ).


## How to use? 🚀

> Documentation under construction.

To use Clair, access our [API documentation](https://next-lab-test.bms.utwente.nl/chatBot/redoc).



## Development ⚡

Clair has three main modules, `app`, `nlu`, and `triggering`.

`app`: Module that implements the API.

`nlu`: Module that implements the dialogue variables and data parsing.

`triggering`: Module that implements the triggering mechanism and agent manager to select talk moves using a fuzzy expert system.


### Installation steps for deploying app with Docker 🐳

Build the container
```
docker build --no-cache -t clair:<version> -f Dockerfile .
```

Setup mongo-db
```
docker run --name mongo-db -d mongo:5.0
docker exec -it mongo-db mongo admin --eval "db.createUser({ user: 'admin', pwd: '<your_pwd>', roles: ['root'] });"
```

Run the containers of clair and mongo-db
```
docker-compose up -d
```

Check the logs
```
docker logs -f <container_id>
```

Check if the app is running by accessing the following URL:
```http://<host>:8000/redoc```

### Installation steps to deploy locally 🛠️

Create conda env and install requirements using the following steps:

```
conda create -n clair python=3.9
conda activate clair
pip install -r requirements.txt
```

```
python -m uvicorn app.app:app --host 0.0.0.0 --port 8000
```

Check if the app is running by accessing the following URL:
```http://localhost:8000/redoc```


### Tests 🧪

When developing Clair, it is recommended to do some extensive tests with your 
topic and language to see if everything runs as expected. 

Unit tests can be executed as follows:
```
python tests/test_app.py (requires the app to be running locally)
python tests/test_nlu.py
python tests/test_triggering.py
```

Overall functionality of Clair can be tested with `tests/clair_api_tester.py`. 

You will need to have a dataset (`.csv` file) of student dialogue with the following attributes.

* learning_space
* group
* username
* timestamp
* text

While Clair API already running, use `tests/clair_api_tester.py` as follows:
```
set CLAIR_URL=http://localhost:8000
python .\tests\clair_api_tester.py --data_file_path=<path>\<your_file>.csv --n_groups=2
```

The script sends all the messages to the API and collect the dialogue variables and agent 
interventions, exporting the results to an excel file.


## License 📜

Clair is an intellectual property from the University of Twente. 

Distribution or usage for commercial purposes without the consent of the University of Twente is not permitted.

> License under construction... 


## Resources 📚

For more info on Clair, check out our latest papers: 

- [Enhancing student dialogue productivity with learning analytics and fuzzy rules (AIED'24)](https://adaj.github.io/files/Clair-AIED24.pdf)
- [A learning analytics-based collaborative conversational agent to foster productive dialogue in inquiry learning (JCAL)](https://doi.org/10.1111/jcal.13007)


## Get involved 💬

Join the Clair community on [Discord](https://discord.gg/kS3t75pkFZ).
