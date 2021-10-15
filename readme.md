# An Examination of Vaccination and Covid-19  in India
# How to Recreate this Work
Data files can be found at the following link: <br>
https://www.kaggle.com/sudalairajkumar/covid19-in-india?select=covid_19_india.csv <br>
Download the following csv's: <br>
covid_19_india.csv <br>
covid_vaccine_statewise.csv <br>
The csv's are also available in the github repository. <br> <br>
From github copy: <br>
covid_india_project_final.ipynb <br>
helper_file.py <br> 
Github repository link:<br>
https://github.com/ian-james-johnson/individual_project <br> <br>
All files should be placed in the same directory. <br>
Open the jupyter notebook. <br>
In the drop-down menu; select 'Cell' and then 'Run All'
# Executive Summary
### Problem: 
Covid-19 is causing great harm to India.
### Solution: 
Analysis of Covid-19 data across states can identify best/worst responding states, which can be used to improve vaccination, treatment, and prevention protocols. <br> <br>
What one state is doing right can be transferred to other states. <br> <br>
What one state is doing wrong can be prevented in other states.
### Key Findings:
The state in India has tremendous impact on the number of people cured of Covid-19, the number of people killed by Covid-19, and the number of adverse reactions to vaccines (AEFI). <br> <br>
Of the three states examined: <br>
- Delhi is poor at infection spread and AEFI
- Tamil Nadu is poor at infection spread and good at preventing AEFI
- Madhy Pradesh is good at limiting infection spread but poor at AEFI
# Data Source
Covid-19 in India <br>
Dataset on Novel Corona Virus Disease 2019 in India <br>
Author: Sudalairaj Kumar <br>
Data acquired from Kaggle: <br>
https://www.kaggle.com/sudalairajkumar/covid19-in-india?select=covid_19_india.csv <br>
<br>
Files acquired:
- covid_19_india.csv
- covid_vaccine_statewise.csv
- StatewiseTestingDetails.csv
# Data Dictionary
- state: the state in which observations were recorded.
- date: the date at which observations were reported (not necessarily when observations occurred).
- cured: the reported number of patients cured of Covid-19.
- deaths: the reported number of patients killed by Covid-19.
- total_doses: the total doses of all Covid-19 vaccines administered.
- covaxin: the doses of covaxin administered.
- covishield: the doses of covishield administered.
- sputnik: the doses of sputnik administered.
- AEFI: the reported number of adverse reactions to Covid-19 vaccines.
- young_adults_vaccinated: 18-44 yr olds vaccinated.
- midaged_vaccinated: 45 yr olds vaccinated.
- elderly_vaccinated: 60+ yr olds vaccinated.
- males_vaccinated: number of males vaccinated.
- females_vaccinated: number of females vaccinated.
