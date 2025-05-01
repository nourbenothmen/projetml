from folder.nlp_utils import NLPProcessor

processor = NLPProcessor()
test_text = "Ceci est un texte français pour tester le tokenizer."
print(processor.preprocess(test_text))