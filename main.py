import os
import time
from datetime import datetime
import json

import openai
from tqdm import tqdm


timestamp = 1625309472.357246
# convert to datetime
date_time = datetime.fromtimestamp(time.time())
str_date_time = date_time.strftime("%d%m%Y-%H%M%S")

openai.api_key = "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"
imagenet_classes_json = 'map_label.json'

outdir = './outputs'
if not os.path.exists(outdir):
    os.mkdir(outdir)
out_json = os.path.join(outdir, f"imagenet_prompting_{str_date_time}.json")
outstat_json = os.path.join(outdir, f"imagenet_prompting_stats_{str_date_time}.json")

category_list = json.load(open(imagenet_classes_json, 'r'))
all_responses = {}
vowel_list = ['A', 'E', 'I', 'O', 'U']

def preparing_prompts():
    """
    what colors, shapes, textures, sizes, and any other visual appearances can we see from a platypus?

    How would we describe a platypus in a scene?

    what could a platypus be seen with?

    what are the activities a platypus is associated with?

    describe what is it like to be a platypus.
    """
    prompts = []
    prompts.append("What colors could be seen from %s %s?")
    prompts.append("What shapes could be seen from %s %s?")
    prompts.append("What textures could be seen from %s %s?")
    prompts.append("Describe visual appearances of %s %s.")
    prompts.append("Describe %s %s in a scene.")
    prompts.append("What could %s %s be seen with?")
    prompts.append("What are the activities %s %s would be associated with?")
    prompts.append("Describe what is it like to be %s %s.")

    return prompts

all_prompts = preparing_prompts()
stats = {'counts': {}}

count = 0
for category_dict in tqdm(category_list.items()):
    category = category_dict[1]
    if category[0] in vowel_list:
        article = "an"
    else:
        article = "a"

    prompts = [p % (article, category) for p in all_prompts]

    all_results = []
    for curr_prompt in prompts:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=curr_prompt,
            temperature=.99,
            max_tokens=50,
            n=5,
            stop="."
        )
        for response in response["choices"]:
            result = response["text"]
            result = result.replace("\n\n", "") + "."
            if len(result) > 4:     # avoid short response
                all_results.append(result)

    all_responses[category_dict[0]] = all_results

    stats['counts'][category_dict[0]] = len(all_results)

    if count == 0:
        break
    count += 1

print(all_responses)
stats['#category_counts'] = len(all_responses.items())
stats['#total_answer_counts'] = sum(stats['counts'].values())
stats['#avg_answer_counts'] = stats['#total_answer_counts'] / stats['#category_counts']

with open(out_json, 'w') as f:
    json.dump(all_responses, f, indent=4)
with open(outstat_json, 'w') as f:
    json.dump(stats, f, indent=4)