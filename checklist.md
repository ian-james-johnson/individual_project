# ReadMe
Does it include the project description and goals of the project? <br>
Additional comments re: project description and goals? <br>
Does the readme include a useful data dictionary? <br>
Additional Comments re: data dictionary? <br>
Does the readme include a project plan? <br>
Additional Comments re: project plan? <br>
Does the readme include initial ideas/hypotheses? <br>
Additional comments re: initial ideas/hypotheses <br>
Does the readme include instructions on how to reproduce this project? <br>
Clone the repo, and walk through the steps indicated to set up an environment that can reproduce the results. Did you run into any issues in doing this? Did you get stuck somewhere that you were unable to work through? <br>
Additional comments on reproducing the project? <br>
Overall, how would you rate the effectiveness of the readme? <br>
# Helper Modules for Acquiring and Preparing
Does the student include either a wrangle.py file or both acquire.py and prepare.py files? <br>
For each function in the files, are docstrings included? <br>
Is it clear what each function is doing (via code + comments + docstrings)? <br>
Using your cloned repo, open a new notebook, set up the environment with imports and data in the working directory, and run each function individually to test its functionality. Is it running without error and returning what is expected? <br>
Now, go to the student's final notebook to evaluate their helper functions. Are the modules imported into the environment? (if only prepare.py is imported, but acquire.py is imported in prepare.py, then that counts as a yes!) <br>
Are the necessary functions called in order to acquire/prepare the data? (remember, the acquire function may be included in the prepare function, and if that is the case, you won't see it run in the notebook, but you can check the prepare function to see if it references the acquire function. In that case, the answer here would be "YES". <br>
Does acquisition and preparation succeed as expected? <br>
Is there Markdown language in the notebook guiding the reader through what is going on, and why certain decisions were made, like the decision on how to handle missing values? <br>
Are there comments in the code in the acquisition and prep sections explaining what the code is doing? <br>
Are you able to replicate acquisition and preparation and end up with a dataframe that resembles the dataframe in the student's final notebook at that point? <br>
If the answer is not yes, why was it not able to be replicated? <br>
# Splitting the data: train, validate, test
Is the data split into train, validate & test at an appropriate time? (i.e. at some point prior to plotting and comparing the interaction of variables). <br>
If scaling is performed, was is fit on training and transformed on train, validate, and test? <br>
If missing values were imputed using mean, median, or some other method, (not including filling missing values with a standard value, like 0, e.g.), was this fit on train and transformed on train, validate & test? <br>
Were any plots created that used 2 or more variables prior to splitting data in train, test, validate? <br>
Are there comments in the code in the "splitting the data" section explaining what the code is doing? <br>
Any additional feedback on the "splitting the data section"? <br>
# Exploration
Is univariate exploration performed across all the variables? (categorical vars: frequency table/bar charts | numeric vars: histograms/summary statistics (.describe()) <br>
Are you able to reproduce the charts/stats from the code used in univariate exploration? <br>
Are takeaways clearly documented? <br>
Is bivariate exploration performed comparing all independent variables with the dependent variable using at least one of the following methods for each interaction: visualization or appropriate statistical test. <br
Are you able to reproduce the charts/stats from the code used in bivariate exploration? <br>
Are takeaways clearly documented <br>
Is multi-variate exploration performed by visualizing 2 independent variables with 1 dependent variable? And/or subsetting your sample and running a statistical test? <br>
Are you able to reproduce the charts/stats from the code used in multivariate exploration? <br>
Are takeaways clearly documented <br>
Did the student ask one or more specific questions they then proceeded to answer via visualization and/or statistical tests? Did the student give a clear answer to the question in "natural language" and takeaway? <br>
Are you able to reproduce the charts/stats from the code used in answering specific questions? <br>
Is there Markdown language in the notebook guiding the reader through what is going on, takeaways, and why certain decisions were made in the exploration section? <br>
Are there comments in the code in the exploration section explaining what the code is doing? <br>
# Features
Did the student make decisions and document those decisions about including/excluding features based on discoveries found during exploration or feature selection? <br>
Do those decisions seem to be supported by the outcome of exploration, statistical testing, or feature selection algorithms? <br>
Did the student create new features by combining, separating, or otherwise transforming existing feature(s)? <br>
Did the student create dummy variables where they were needed? (all categorical, non-ordered variables that are going to be used in modeling). <br>
Were all numeric variables scaled? <br>
Are you able to reproduce the scaling of features, creating dummy variables, and all other feature adaptations? <br>
# Modeling
At least 3 different models built, using training only!<br>
Models were evaluated on train first, and conclusions were drawn.<br>
The top x models were evaluated on validate, and conclusions were drawn.<br>
The final model was evaluated on test, and conclusions were drawn.<br>
Is there Markdown language in the notebook guiding the reader through what is going on, takeaways, and why certain decisions were made in the modeling section?<br>
Are there comments in the code in the modeling section explaining what the code is doing? <br>
Are you able to reproduce the modeling from the code used in answering specific questions? <br>
# Conclusion
Is there a clear conclusion section that sums up what was accomplished, what was discovered, and next steps? (similar to a conclusion slide in a presentation). Explain and give feedback!
