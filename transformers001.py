from transformers import pipeline
import pandas as pd

fh = open('./text.txt', 'r')
text = ''
for ll in fh:
    text += ll[:-1]

classifier = pipeline("text-classification")
outputs = classifier(text)
result = pd.DataFrame(outputs)

print(result)

