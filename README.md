# Machine Learning in Python, scikit-learn classification

This is my project for Udacity's Intro to Machine Learning class, Identify Fraud in the Enron Dataset, part of Udacity's Data Analyst nanodegree, which is part of Western Governor University's Bachelor of Science in Data Management and Data Analytics.

The entire process can be run from 'poi_id.py' and is explained in 'Free-Response Questions.ipynb', but I have included supplemental materials and other resources.

My first true ML project, it's pretty messy. I wouldn't consider this (or any of my school projects thus far) a finished deliverable.

## Files

- */data*: Contains the starting dataset pickled, and several pickled dictionaries of performance metrics created during the algorithm selection and tuning process not carried out in the final script.
- */supplemental_material*: Contains (most of) the (messy) notebooks I used along the way to explore and experiment with the data and the ML process itself. While I don't recommend running these notebooks, they are there to show my work and how I thought. They should be viewed in this order: 'initial_wrangle.ipynb', 'handling_eda_etc.ipynb', 'feature_engineering.ipynb', 'selection.ipynb', followed by the gridsearch notebooks.
- *Free-Reponse Questions.ipynb*: A notebook with Udacity's questions regarding the project and my process, with my reponses. While Udacity asked for shorter reponses, the extent to which I took the project warranted longer responses in order to address each point of each set of questions and their associated [rubric items](https://review.udacity.com/#!/rubrics/27/view).
- *poi_id.ipynb*: A notebook of final script from which all cleaning, engineering, tuning, validation, and evaluation is run. It breaks up output for easier reference.
- *enron61702insiderpay.pdf*: PDF of financial data with footnotes, from FindLaw.com.
- *environment.yml*: The conda environment I used.
- *Free-Response Questions.html*: HTML of 'Free-Reponse Questions.ipynb'.
- *my_classifier.pkl*: A final (not best) classifier model. It's a scikit-learn pipeline containing a tune feature selection algorithm and a tuned classifier.
- *my_dataset.pkl*: The dataset (as a dictionary) with the features to be plugged into the above model. It includes 'poi' which is the target feature.
- *my_features_list.pkl*: The list of the features in my_dataset.
- *poi_id.py*: The final script from which all cleaning, engineering, tuning, validation, and evaluation is run.

## References

### Imports etc.:
* https://www.python.org/ For all coding.
* https://www.anaconda.com/ For conda and Jupyter Notebooks.
* http://www.picklesdoc.com/ For serializing objects.
* https://pandas.pydata.org/ For dataframes and series'.
* https://numpy.org/ For arrays and stats.
* https://matplotlib.org/ For plotting EDA.
* https://seaborn.pydata.org/ For plotting EDA.
* https://www.scipy.org/ For EDA.
* https://sklearn.org/ For the extensive ML suite.
* https://docs.python.org/3/library/functools.html/ For callables with set parameters.
* https://docs.python.org/3/library/time.html For timing operations.
* https://ipython.org/ For the interactive shell in Jupyter Notebook.
* https://docs.python.org/3/library/sys.html For setting path.
* https://udacity.com/ For the educational material, starter code, and preprocessed data.

Data:
* https://www.cs.cmu.edu/~enron/ email dataset
* https://findlaw.com/ The financial dataset. Udacity provided this, and I have been unable to find it on findlaw.com, but I included the PDF in the supplemental material folder for reference.
* https://udacity.com/ For the educational material, starter code, and preprocessed data.


Other people's approaches:
* https://medium.com/@Tushar_007/analysis-of-financial-data-of-enron-8457df24b6af
* https://williamkoehrsen.medium.com/machine-learning-with-python-on-the-enron-dataset-8d71015be26d
I read these writeups to see how others have approached the same problem. Though I didn't borrow any code, nor ideas that aren't already common, William Koehrsen's article reminded me to validate the data against the total columns, and reading his explanation saved me the trouble of puzzling out why there were errors.


General information about the scandal and the data.
* https://enrondata.readthedocs.io/en/latest/
* https://foreverdata.org/1009HOLD/Enron_Dataset_Report.pdf
* "Enron: The Smartest Guys in the Room," 2005 documentary available on Netflix.


Education/reference:
* https://udacity.com/ For the educational material, starter code, and preprocessed data.
* "Data Skeptic" podcast. Early episodes introduced me to key concepts, especially regarding information leakage and the multiple comparisons problem.
* https://en.wikipedia.org/wiki/Multiple_comparisons_problem
* https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf For suggested C and gamma search ranges in SVMs.
* https://stackoverflow.com/users/2391771/kaleb-coberly For troubleshooting (no copied code other than below).
* https://stackoverflow.com/questions/2613800/how-to-convert-dos-windows-newline-crlf-to-unix-newline-lf-in-a-bash-script/19702943#19702943 Adapted @anatoly techtonik's solution for unpickling objects created by Unix. This is a commonly copied/pasted script. See crlf_to_lf in 'data/doc2unix.py'.
