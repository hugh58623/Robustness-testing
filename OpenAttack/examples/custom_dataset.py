'''
This example code shows how to use the PWWS attack model to attack BERT on a customized dataset.
'''
import OpenAttack
import transformers
import datasets
import  ssl
ssl._create_default_https_context = ssl._create_unverified_context    
def main():
    # load a fine-tuned sentiment analysis model from Transformers (you can also use our fine-tuned Victim.BERT.SST)
    print("Load model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(r"D:/project/freework/db_attack/data/Victim.BERT.SST")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(r"D:/project/freework/db_attack/data/Victim.BERT.SST", num_labels=2, output_hidden_states=False)
    victim = OpenAttack.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)

    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker()

    # create your dataset here
    dataset = datasets.Dataset.from_dict({
        "x": [
            "I hate this movie.",
            "I like this apple."
        ],
        "y": [
            0, # 0 for negative
            1, # 1 for positive
        ]
    })

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, victim, metrics = [
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ])
    attack_eval.eval(dataset, visualize=True)

if __name__ == "__main__":
    main()
