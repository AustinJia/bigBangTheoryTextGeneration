This is a course project for CSE 6240.

colab with edit permission:
https://colab.research.google.com/drive/1EHPSoLq1u_1eD2dmO5eIpH1SnC2zdADm

data.pkl stores the dictionary: mapping from series_number to content

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

Code based on following repositories:
- https://codeburst.io/web-scraping-101-with-python-beautiful-soup-bb617be1f486
- https://github.com/udacity/deep-learning/tree/master/tv-script-generation
- https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-tv-script-generation
- https://github.com/koushik-elite/TV-Script-Generation
