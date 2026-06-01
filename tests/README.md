# Tests for `clair`

This directory contains usage and unit tests for various components of the project. Ensure all the tests pass before making a commit or deploying.


## Usage tests

- Ensure that the `clair` server is running on `localhost:5000`.
- Run the unittests: `python tests/test_<module>.py` (for `app/`, `nlu/` and `triggering/`).
- Adjust parameters (or add code) as needed to simulate different scenarios.


### 1. `clair_api_tester.py`

This script is designed to interface with the Clair API to test data interactions.
Given a specified CSV dataset, it will send data to the Clair API and capture 
the responses in an Excel spreadsheet.

This does not use python's unittest as it's more of an auxiliary tool than an actual unit testing script.

The data required to run this script is a CSV file with at least the following columns:
    - group: the group of the message
    - username: the username of the message
    - timestamp: the timestamp of the message
    - text: the text of the message

```
    cd tests/
    set (or export, on Linux) CLAIR_URL=<YOUR_CLAIR_URL>
    python clair_api_tester.py --data_file_path=<path_to_data_file> --n_groups=1
```
    

### 2. `clair_api_debug.ipynb`

This is an auxiliary tool for debugging the API within a jupyter environment for more interactivity. As `clair_api_tester`, it can be used to send requests to the API and visualize responses, helping in manual testing. This does not use python's unittest as it's more of an auxiliary tool than an actual unit testing script.


## Unit tests

### 1. `test_nlu.py`

Unittest of `nlu/`, based on a few representative scenarios.

**Key Components Tested**:
- **Parsing**: Checks if the parsing functionality is correct.
- **ConSent**: Checks if the variables are correctly measured.

### 2. `test_triggering.py`

Unittest of `triggering/`, based on a few representative scenarios.

**Key Components Tested**:
- **Rule Parsing**: Checks if translation of rules into a format suitable for the fuzzy logic engine is correct.
- **Triggering Mechanism**: Checks if  given a dialogue state, the system correctly computes the fuzzy output.
- **Agent Manager**: Checks the functioning of the agent manager, ensuring that the correct agent is triggered for a given dialogue state.

### 3. `test_app.py`

Unittest of `app/`, based on a few representative scenarios.

**Key Components Tested**:
- **API Endpoints**: Verifies the correctness of API responses for given inputs.
- **Session Handling**: Validates the app's ability to handle sessions and maintain context over multiple interactions.

