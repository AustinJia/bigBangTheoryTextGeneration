
## Text Generation for The Big Bang Theory (TBBT)
This is a course project for CSE 6240: Web Search and Text Mining, Spring 2020

###### Authors: Raghav Raj Mittal, Austin Jiang, Satwik Mekala
[link to Our Github](https://github.com/AustinJia/bigBangTheoryTextGeneration).

1.	Scrapping code in colab (recommend) with edit permission. [Link to scrapping_tools.ipynb!](https://colab.research.google.com/drive/1EHPSoLq1u_1eD2dmO5eIpH1SnC2zdADm)

	Or You can run scrapping_tools.py

2.	Model generator code in colab with edit permission. [Link to model_generator.ipynb!](https://colab.research.google.com/drive/1PJJt6lzVxeGxMv4brBLiElOTP62ILQ8K)

	Before the running make sure 
- Make sure select GPU type in colab (Runting->Change Run Type -> GPU)
- Make sure upload following files (models.py torch_utils.py helper.py preprocess.pkl)

3.	Text generator code in colab with edit permission. [Link to text_generator.ipynb!](https://colab.research.google.com/drive/1GrpRP17WUOQlW9WbBuRN0izqqPhd2iYY)
	Before the running make sure 
- Make sure select CPU type in colab (Runting->Change Run Type -> None)
- Make sure upload following file before run, helper.py preprocess.pkl models.py data.pkl trained_GRU_sq_3.pt(this file could be different , and also change "model_name") 

4. Code expalining

| Code  | Definition |
| ------------- | ------------- |
| scrapping_tools.ipynb  |  scrapping code to get TBBT data |
| model_generator.ipynb  |  generate model |
| text_generator.ipynb  | generate text based on different input model  |
| word2vecTrainModels.py  | word2vec training model  |
| generator_bleu_test.ipynb  | bleu score for all models once created  |


5.	data.pkl,
	stores the dictionary: mapping from series_number to content

```
dict = {
  "series-1-episode-1": "['Scene: A corridor at a sperm bank.', 'Sheldon: So if a photon is directed through a plane with two slits in it and either slit is observed it will not go through both slits. If it’s unobserved it will, however, if it’s observed after it’s left the plane but before it hits its target',...]",
  "series-1-episode-2": ...,
  "series-1-episode-3": ...,
  ...
  'series-10-episode-23':...,
  'series-10-episode-24:...'
}
```

You also can print out
```
print(dict.keys())# to see all the keys
```

6. In preprocess.pkl, it stores the preporecessed data.

7. Code based on following repositories:
- https://codeburst.io/web-scraping-101-with-python-beautiful-soup-bb617be1f486
- https://github.com/udacity/deep-learning/tree/master/tv-script-generation
- https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-tv-script-generation
- https://github.com/koushik-elite/TV-Script-Generation

