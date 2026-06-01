# data

Documentation under construction...

## Generating training data prompting LMs

Here NEGATIVE VALENCE MONITORING COGNITION is the example of intent we are looking to detect in upcoming text. An example with "but not CONFUSION" can be helpful if you need to distinguish between two similar intents and focus on the one wanted.

```prompt1
Generate 20 samples of informal whatsapp-like messages from two learners in a collaborative science lesson particularly expressing NEGATIVE VALENCE MONITORING COGNITION but not CONFUSION. Include in some of the samples words like "I/we don't/can't/... ". These messages should contain one sentence only. I will train an ML model using it so try to be as short as possible in the sentences (maybe 5-7 words maximum), it might be better for model generalization. These messages do not have to be sequential, they can be messages that may happen at any point of the dialogue. 
```

```prompt2
Now twenty similar (in length also) samples of POSITIVE VALENCE MONITORING COGNITION.
```
    
```prompt3
Now twenty similar (in length also) samples of NEUTRAL VALENCE MONITORING COGNITION.
```

