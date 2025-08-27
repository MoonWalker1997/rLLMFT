_truth_categories = [
    (0.0, 0.2, ["is false", "completely false", "does not hold", "has been refuted", "is incorrect"]),
    (0.2, 0.4, ["is mostly false", "tends to be false", "seems incorrect", "barely holds", "is generally wrong"]),
    (0.4, 0.6,
     ["is unknown", "is undetermined", "cannot be classified", "its truth is unclear",
      "is neither true nor false"]),
    (0.6, 0.8, ["is mostly true", "tends to be true", "seems correct", "largely holds", "is generally valid"]),
    (0.8, 1.01, ["is true", "completely holds", "has been confirmed", "is correct", "is a fact"])
]

_inheritance_templates = [
    "{sub} is a type of {obj}",
    "Every {sub} is an instance of {obj}",
    "{sub} falls under the category of {obj}",
    "{sub} can be seen as a specialization of {obj}",
    "{obj} generalizes the concept of {sub}",
    "{obj} includes all instances of {sub}",
    "If something is a {sub}, then it is a {obj}",
    "{sub} belongs to the broader class of {obj}",
    "{sub} is more specific than {obj}",
    "{sub} is a manifestation of {obj}",
    "{obj} is a superclass of {sub}",
    "{sub} derives from the category {obj}",
    "{sub} is subsumed by {obj}",
    "{sub} should be classified under {obj}",
    "{sub} expresses all attributes of {obj}",
    "In the context of {obj}, {sub} is a specific case",
    "{sub} is an instantiated form of {obj}",
    "{sub} is a narrower subtype of {obj}",
    "What we call {sub} is just a kind of {obj}",
    "Part of what makes up {obj} is represented by {sub}"
]

_inheritance_templates_q = [
    "Is {sub} a type of {obj}",
    "Is every {sub} an instance of {obj}",
    "Does {sub} fall under the category of {obj}",
    "Can {sub} be seen as a specialization of {obj}",
    "Does {obj} generalize the concept of {sub}",
    "Does {obj} include all instances of {sub}",
    "If something is a {sub}, then is it a {obj}",
    "Does {sub} belong to the broader class of {obj}",
    "Is {sub} more specific than {obj}",
    "Is {sub} a manifestation of {obj}",
    "Is {obj} a superclass of {sub}",
    "Does {sub} derive from the category {obj}",
    "Is {sub} subsumed by {obj}",
    "Should {sub} be classified under {obj}",
    "Does {sub} express all attributes of {obj}",
    "In the context of {obj}, is {sub} a specific case",
    "Is {sub} an instantiated form of {obj}",
    "Is {sub} a narrower subtype of {obj}",
    "Is what we call {sub} just a kind of {obj}",
    "Is part of what makes up {obj} represented by {sub}"
]

_similarity_templates = [
    "{sub} and {obj} are conceptually identical",
    "{sub} is the same as {obj}",
    "{sub} and {obj} refer to the same thing",
    "{sub} equals {obj}",
    "{sub} and {obj} are interchangeable terms",
    "{sub} and {obj} describe the same category",
    "Whether you say {sub} or {obj}, it means the same",
    "{sub} is also known as {obj}",
    "{obj} is an alternative name for {sub}",
    "{sub} and {obj} are equivalent concepts",
    "{sub} and {obj} have no distinction in meaning",
    "People consider {sub} and {obj} to be the same",
    "{sub} and {obj} can substitute for each other",
    "{sub} and {obj} are synonyms",
    "To us, {sub} and {obj} have the same definition",
    "Both {sub} and {obj} signify the same thing",
    "{sub} is recognized as equivalent to {obj}",
    "{sub} and {obj} mutually define one another",
    "{sub} is a valid replacement for {obj}",
    "{sub} and {obj} share a bidirectional ontological relation"
]

_similarity_templates_q = [
    "Are {sub} and {obj} conceptually identical",
    "Is {sub} the same as {obj}",
    "Do {sub} and {obj} refer to the same thing",
    "Does {sub} equal {obj}",
    "Are {sub} and {obj} interchangeable terms",
    "Do {sub} and {obj} describe the same category",
    "Whether you say {sub} or {obj}, does it mean the same",
    "Is {sub} also known as {obj}",
    "Is {obj} an alternative name for {sub}",
    "Are {sub} and {obj} equivalent concepts",
    "Do {sub} and {obj} have no distinction in meaning",
    "Do people consider {sub} and {obj} to be the same",
    "Can {sub} and {obj} substitute for each other",
    "Are {sub} and {obj} synonyms",
    "To us, do {sub} and {obj} have the same definition",
    "Do Both {sub} and {obj} signify the same thing",
    "Is {sub} recognized as equivalent to {obj}",
    "Do {sub} and {obj} mutually define one another",
    "Is {sub} a valid replacement for {obj}",
    "Do {sub} and {obj} share a bidirectional ontological relation"
]

cs = ["MP, SM", "PM, SM", "M<>P, SM", "MP, MS", "PM, MS", "M<>P, MS", "MP, S<>M", "PM, S<>M", "M<>P, S<>M"]
