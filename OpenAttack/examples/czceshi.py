'''
This example code shows how to use the PWWS attack model to attack BERT on a customized dataset.
'''
import OpenAttack
import transformers
import datasets
import  ssl
ssl._create_default_https_context = ssl._create_unverified_context    

import json

        
def main(ori_file_path):
    # load a fine-tuned sentiment analysis model from Transformers (you can also use our fine-tuned Victim.BERT.SST)
    print("Load model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(r'/home/yangliu6/scratch/OpenAttack-xiugai/Models/distilbert-base-uncased/')
    model = transformers.AutoModelForSequenceClassification.from_pretrained(r'/home/yangliu6/scratch/OpenAttack-xiugai/Models/distilbert-base-uncased/', output_hidden_states=False)
    victim = OpenAttack.classifiers.TransformersClassifier(model, tokenizer, model.distilbert.embeddings.word_embeddings)

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker()

    my_sen = []
    with open(ori_file_path) as file:
        for line in file:
            json_obj = json.loads(line)
            second_field = json_obj.get("prompt")
            my_sen.append(second_field.replace('    ','').replace('\n','').replace('\"','')[:512])
        
    # create your dataset here
    dataset = datasets.Dataset.from_dict({
        "x": my_sen
    })

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, victim,metrics = [
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ])
    attack_eval.eval(ori_file_path, dataset, visualize=True)

if __name__ == "__main__":
    ori_file_path = 'D:/project/freework/db_attack/humaneval_partial.jsonl'
    main(ori_file_path)
