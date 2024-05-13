import pylab as plt
import json
import pandas as pd 
json_file = '/group/dl4miacourse/pokemon/data-chad/First40training_last10validation/First40_training_annotations/labels_first40_2023-10-17-12-04-26.json'
file =  open(json_file,'r')

json_data = json.load(file)

print(pd.json_normalize(json_data['annotations']))
plt.plot(pd.json_normalize(json_data['annotations'])['category_id'],'.')

plt.show()