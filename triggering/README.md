# triggering

Before running the app, make sure you run the triggering setup and check your `rules.yml` file.

How to configure the fuzzy memberships?
```
# Example of mode APT-base
python triggering/memberships.py inputs --mode="apt-base"
python triggering/memberships.py outputs --mode="apt-base"
python triggering/fuzzy.py plot_membership --mode="apt-base"
```

How to test the module?
```
python tests/test_triggering.py
python tests/test_manager.py
```

For a deeper reference on fuzzy expert systems, please check out 
http://www.computing.surrey.ac.uk/ai/PROFILE/fuzzy and 
https://scikit-fuzzy.readthedocs.io/en/latest/.


## More info

> Documentation under construction.