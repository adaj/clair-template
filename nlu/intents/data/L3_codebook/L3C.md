## Codebook for Confusion/Certainty (L3C) Classification

### 1. Confusion
**Criteria:**
   - Expresses a lack of understanding or clarity.
   - Indicates uncertainty, doubt, or difficulty in grasping a concept.
   - May include direct questions seeking clarification.
   - Often uses phrases like "I don't understand", "I'm not sure", "This is confusing", "Can you explain", etc.
   - Includes expressions that show the speaker is lost, puzzled, or finding something complex.

**Examples:**
   - "I'm struggling to grasp this idea."
   - "Why is it done this way?"
   - "I'm not getting it."

### 2. Certainty
**Criteria:**
   - Conveys a strong sense of understanding or assurance.
   - Indicates confidence in knowledge or understanding.
   - Uses phrases like "I'm confident", "I'm sure", "I know", "There's no doubt", etc.
   - Can include expressions of personal conviction or assertions of fact.
   - May include references to personal study, knowledge, or experience that back up the certainty.

**Examples:**
   - "I'm positive that this is the way."
   - "I understand this clearly."
   - "I've studied this thoroughly."

### 3. Neutral Statement
**Criteria:**
   - Does not explicitly express confusion or certainty.
   - Can be informational, procedural, or observational in nature.
   - Often includes factual statements, general inquiries, summaries, or neutral clarifications.
   - May involve restating information or setting the stage for discussion without showing personal understanding or lack thereof.
   - Includes phrases that are more about logistics, schedules, resources, or neutral acknowledgments.

**Examples:**
   - "The next step involves analyzing the data."
   - "This section covers the basics of the theory."
   - "The group project is due next month."

### Guidelines for Coders:

1. **Context Consideration:** Always consider the context of the sentence. The same phrase can have different meanings in different contexts.

2. **Look for Key Phrases:** Use the listed criteria and examples as a guide to identify key phrases that typically indicate confusion, certainty, or neutrality.

3. **Consistency is Key:** Apply these criteria consistently across all data to maintain reliability in coding.

4. **When in Doubt, Choose Neutral:** If a sentence could reasonably fit into more than one category or if the intent is not clear, classify it as a neutral statement.


### Some edge cases

```
[
    "This seems correct, though I'm curious about alternative methods.",
    "I'm not completely lost, but this part is a bit complex for me.",
    "Most of this makes sense, except for one small detail.",
    "I've almost got it, just need to clarify one point.",
    "The theory is clear, yet applying it seems challenging.",
    "I understand you, but I'm not entirely sure why each step is necessary.",
    "That explanation seems clear, but does it apply in all cases?",
    "I agree with the concept, though I haven't seen it applied practically.",
    "This part of the lecture was straightforward, yet I wonder how it relates to our project.",
    "I see what the author is getting at, but isn't there an exception to this theory?"
]
```